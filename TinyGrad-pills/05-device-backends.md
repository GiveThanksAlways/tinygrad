# Pill 5: Device Backends & the Hardware Command Queue

## The Big Picture

TinyGrad abstracts over multiple GPU vendors through a single framework called **HCQ** — Hardware Command Queue. Instead of going through CUDA Runtime or OpenCL, HCQ talks directly to the kernel driver via ioctls:

```
┌─────────────────────────────────────────────────┐
│              TinyGrad Engine                      │
│  (schedule → optimize → compile → execute)       │
└────────────┬────────────────────────────────────┘
             │  HCQ API (abstract interface)
┌────────────┴────────────────────────────────────┐
│  HCQCompiled  │  HWQueue  │  HCQAllocator       │
│  HCQProgram   │  HCQSignal│  HCQBuffer           │
└──┬─────────┬──┴───────┬───┴──────────────────────┘
   │         │          │
┌──┴──┐  ┌──┴──┐   ┌──┴──┐   ┌──────┐
│ NV  │  │ AMD │   │QCOM │   │ CPU  │
│ops_ │  │ops_ │   │ops_ │   │ops_  │
│nv.py│  │amd. │   │qcom.│   │cpu.  │
│     │  │py   │   │py   │   │py    │
└──┬──┘  └──┬──┘   └──┬──┘   └──────┘
   │        │         │
 nvidia   amdgpu    adreno
 .ko /    .ko       .ko
 nvgpu
 .ko
```

Every GPU backend in TinyGrad implements the same abstract interface. Your tensor code doesn't know (or care) whether it's running on an NVIDIA, AMD, or Qualcomm GPU.

## Why Not CUDA Runtime?

CUDA Runtime (`libcudart.so`) is NVIDIA's official API. It's easy to use:

```c
cudaMalloc(&ptr, size);
cudaMemcpy(ptr, host_data, size, cudaMemcpyHostToDevice);
myKernel<<<grid, block>>>(ptr);
cudaDeviceSynchronize();
```

But it's also:
- **Proprietary** — you need NVIDIA's closed-source libraries
- **Opaque** — you can't see or control what happens between `<<<>>>` and hardware
- **Slow for small kernels** — launch overhead can dominate short compute jobs
- **Not portable** — AMD/Qualcomm have their own completely different APIs

TinyGrad's philosophy: **bypass the middleman**. Talk to the kernel driver directly. Build command buffers in userspace. Submit them to hardware queues yourself.

The result? Lower latency, full control, and one abstraction that works across vendors.

## HCQ Anatomy: The Six Core Classes

### 1. HCQCompiled — The Device

`HCQCompiled` is the base class for all HCQ devices. Each physical GPU becomes an `HCQCompiled` instance.

```python
class HCQCompiled(Compiled, Generic[SignalType]):
    # Device identity
    self.device_id: int          # e.g., 0 for "NV:0"
    self.peer_group: str         # e.g., "NV" — groups devices that share memory

    # Timeline synchronization
    self.timeline_value: int     # monotonically increasing counter
    self.timeline_signal: Signal # the signal tracking our timeline

    # Queue constructors
    self.hw_compute_queue_t      # factory for compute queues
    self.hw_copy_queue_t         # factory for copy/DMA queues (optional)

    # Kernel args scratch space
    self.kernargs_buf: HCQBuffer            # big pre-allocated buffer
    self.kernargs_offset_allocator: BumpAllocator  # bump allocator into it
```

**NVDevice** inherits from `HCQCompiled[NVSignal]` and adds NVIDIA-specific init:

```python
class NVDevice(HCQCompiled[NVSignal]):
    def __init__(self, device:str=""):
        # Select interface: TegraIface (Jetson) vs NVKIface/PCIIface (desktop)
        self.iface = self._select_iface(NVKIface, PCIIface, TegraIface)

        # GPU properties from hardware
        self.sm_version = ...   # 0x807 for SM 8.7
        self.gpc_count = ...    # GPU Processing Cluster count
        self.tpc_count = ...    # Texture Processing Cluster count

        # Virtual address space
        self.gpu_va = ...       # VA range for GPU-visible allocations

        # Initialize queues, allocator, compilers
        super().__init__(device, NVAllocator(self), ...)
```

### 2. HWQueue — Command Buffers

`HWQueue` is an in-memory list of commands. You build them up, then submit.

```python
class HWQueue:
    self._q: list    # the actual command words

    def exec(prg, args_state, global_size, local_size): ...  # launch a kernel
    def copy(dest, src, copy_size): ...           # DMA copy
    def signal(signal, value): ...                # write to signal after work finishes
    def wait(signal, value): ...                  # stall until signal >= value
    def timestamp(signal): ...                    # record GPU timestamp
    def memory_barrier(): ...                     # ensure coherence
    def submit(dev): ...                          # push to hardware
```

The key insight: **queues are chainable**:

```python
# Build a compute queue with wait → barrier → exec → signal, then submit
NVComputeQueue() \
    .wait(dev.timeline_signal, dev.timeline_value - 1) \
    .memory_barrier() \
    .exec(program, args, global_size=(256,1,1), local_size=(128,1,1)) \
    .signal(dev.timeline_signal, dev.next_timeline()) \
    .submit(dev)
```

This entire chain builds up a command buffer (a list of 32-bit words) and submits it to the GPU's GPFIFO ring buffer in one ioctl call.

**NVComputeQueue** and **NVCopyQueue** are the NVIDIA implementations:

```python
class NVComputeQueue(NVCommandQueue):
    # Adds GPU-specific commands:
    # - QMD submission for kernel launches
    # - Semaphore release/acquire for signals
    # - L2 cache flush for memory barriers

class NVCopyQueue(NVCommandQueue):
    # Adds DMA engine commands:
    # - Line copy / multi-line copy
    # - No compute — just moves bytes
```

### 3. HCQSignal — GPU/CPU Synchronization

Signals are the synchronization primitive. A signal is a 64-bit value in GPU-visible memory that both CPU and GPU can read/write.

```python
class HCQSignal:
    self.value_addr: int      # GPU VA of the 64-bit value
    self.timestamp_addr: int  # GPU VA of 64-bit timestamp (value_addr + 8)
    self.value_mv: memoryview # CPU access to the value
```

**Timeline protocol**: Every device has a `timeline_signal` and a `timeline_value`. Each submitted command increments the timeline:

```
Timeline: ─────1─────2─────3─────4─────5─────►
               │     │     │     │     │
              Q1    Q2    Q3    Q4    Q5    (submitted work)

CPU waits:  dev.timeline_signal.wait(5)
                                        ↑ blocks until GPU writes 5
```

```python
# GPU side (in the command queue):
queue.signal(dev.timeline_signal, 3)  # "I finished batch 3"

# CPU side:
dev.timeline_signal.wait(3)  # block until signal >= 3 → "batch 3 done"

# Another queue can also wait on GPU:
queue2.wait(dev.timeline_signal, 3)  # "don't start until batch 3 finishes"
```

This is how TinyGrad does **dependency tracking without a runtime**. No CUDA streams, no events — just incrementing integers in shared memory.

### 4. HCQProgram — Compiled Kernels

`HCQProgram` wraps a compiled GPU binary and knows how to launch it.

```python
class HCQProgram:
    self.name: str                    # kernel name
    self.args_state_t: Type           # how to pack arguments
    self.kernargs_alloc_size: int     # bytes needed for arguments

    def fill_kernargs(bufs, vals) -> HCQArgsState:
        # Pack buffer pointers + scalar values into a contiguous region
        ...

    def __call__(*bufs, global_size, local_size, vals, wait):
        # Build queue: wait → barrier → exec → signal → submit
        ...
```

**NVProgram** extends this with ELF parsing, QMD construction, register/shared-memory tracking (see [Pill 4](04-nvidia-gpu.md)).

### 5. HCQBuffer — GPU Memory

```python
class HCQBuffer:
    self.va_addr: int          # GPU virtual address
    self.size: int             # allocation size in bytes
    self.view: MMIOInterface   # CPU-mapped view (if cpu_access=True)
    self.meta: Any             # backend-specific metadata
    self._base: HCQBuffer      # parent buffer (for sub-allocations)
```

Buffers can be **offset** to create sub-views without new allocations:

```python
big_buf = allocator.alloc(1024)                # 1024 bytes at VA 0x1000
sub_buf = big_buf.offset(offset=256, size=128) # 128 bytes at VA 0x1100
```

This is how `kernargs_buf` works — one big allocation, parceled out with a bump allocator.

### 6. HCQAllocator — Memory Management

```python
class HCQAllocator(HCQAllocatorBase):
    def _copyin(dest, src):     # CPU → GPU
    def _copyout(dest, src):    # GPU → CPU
    def _transfer(dest, src):   # GPU → GPU (peer)
    def copy_from_disk(dest, src):  # disk → GPU (streaming)
    def map(buf):               # make buffer visible to this device
```

**The copy pipeline on Jetson is special**. On desktop NVIDIA, `_copyin` uses DMA queues (`NVCopyQueue`) with staging buffers. On Jetson with unified memory, it can use `ctypes.memmove` directly — no DMA needed:

```python
def _copyin(self, dest, src):
    if self.dev.hw_copy_queue_t is None:
        # No copy queue → do direct memcpy (Tegra path)
        self.dev.synchronize()
        ctypes.memmove(int(dest.va_addr), from_mv(src), len(src))
        return

    # Desktop path: stage through bounce buffers via DMA
    for i in range(0, src.nbytes, self.b[0].size):
        self.b_next = (self.b_next + 1) % len(self.b)
        self.dev.timeline_signal.wait(self.b_timeline[self.b_next])
        # ... copy chunk to staging buffer, DMA to GPU ...
```

## How a Kernel Actually Runs

Let's trace the full execution path from `tensor.realize()` to GPU completion:

```
tensor.realize()
    │
    ▼
run_schedule(schedule_items)          # engine/realize.py
    │
    ▼
for item in schedule:
    runner = get_runner(item.ast)     # CompiledRunner (cached by AST)
    runner.exec(item.bufs, item.metadata)
    │
    ▼
CompiledRunner.exec()
    │
    ├── prg = NVProgram(...)          # compiled PTX → CUBIN (cached)
    ├── args = prg.fill_kernargs(bufs, vals)  # pack args → kernargs buffer
    │
    ▼
NVComputeQueue()                      # create command buffer
    .wait(timeline_signal, N-1)       # wait for previous work
    .memory_barrier()                 # flush caches
    .exec(prg, args, global, local)   # encode QMD + launch
    .signal(timeline_signal, N)       # mark completion
    .submit(dev)                      # write to GPFIFO ring buffer
    │
    ▼
# GPU hardware reads GPFIFO entry
# GPU pulls QMD from memory
# SMs fetch SASS binary, launch warps
# GPU writes N to signal memory when done
```

Here's the GPFIFO submission in more detail:

```
┌─────────────────────────────────────┐
│         CPU (Python/ctypes)          │
│                                      │
│  1. Build pushbuffer (NV methods)    │
│  2. Write GPFIFO entry:             │
│     { address: pushbuf_va,          │
│       length: pushbuf_words }       │
│  3. Ring doorbell MMIO              │
│     (write to BAR0 / nvmap region)  │
└──────────────┬──────────────────────┘
               │ doorbell write
               ▼
┌─────────────────────────────────────┐
│            GPU Hardware              │
│                                      │
│  1. Sees doorbell → reads GPFIFO    │
│  2. GPFIFO points to pushbuffer     │
│  3. Pushbuffer contains methods:    │
│     - SET_SHADER_LOCAL_MEMORY       │
│     - LOAD_QMD_BASE_POINTER        │
│     - SEND_QMD                      │
│  4. QMD describes the kernel launch │
│  5. Schedule warps → execute → done │
│  6. Write signal value to memory    │
└─────────────────────────────────────┘
```

## MMIOInterface & FileIOInterface

Two low-level utilities that everything else builds on:

### MMIOInterface: Memory-Mapped Hardware Access

```python
class MMIOInterface:
    def __init__(self, addr, nbytes, fmt='B'):
        self.mv = to_mv(addr, nbytes).cast(fmt)

    def __getitem__(self, k): return self.mv[k]
    def __setitem__(self, k, v): self.mv[k] = v
    def view(offset, size, fmt): return MMIOInterface(...)
```

This wraps a raw memory-mapped address (from `mmap()`) as a typed Python array. Used for:
- **Doorbell registers**: writing a value to trigger GPU attention
- **Signal memory**: reading/writing 64-bit signal values
- **Kernel arguments**: packing buffer pointers and scalar values
- **GPU control registers**: queue configuration

### FileIOInterface: Device File Access

```python
class FileIOInterface:
    def __init__(self, path, flags):
        self.fd = os.open(path, flags)

    def ioctl(self, request, arg): ...    # GPU driver ioctls
    def mmap(self, start, sz, ...): ...   # map GPU memory
    def read(self, size): ...             # read device files
```

This wraps Linux file descriptors for talking to GPU drivers. On desktop NVIDIA, it opens `/dev/nvidia0` and `/dev/nvidiactl`. On Jetson, it opens `/dev/nvhost-gpu` and `/dev/nvmap`.

## Comparing Backends

| Feature | NV (Desktop) | NV (Tegra/Jetson) | AMD | QCOM |
|---------|-------------|-------------------|-----|------|
| Device | NVDevice | NVDevice | AMDDevice | QCOMDevice |
| Interface | NVKIface/PCIIface | TegraIface | KFDIface/PCIIface | — |
| Driver | nvidia.ko | nvgpu.ko + nvmap.ko | amdgpu.ko | adreno |
| Compute Queue | NVComputeQueue | NVComputeQueue | AMDComputeQueue | QCOMComputeQueue |
| Copy Queue | NVCopyQueue | **None** (memmove) | AMDCopyQueue | — |
| VA Space | 48-bit | 40-bit | 48-bit | 32-bit |
| Signal | Semaphore release | Semaphore release | Fence | — |
| Command Format | NV methods (pushbuf) | NV methods (pushbuf) | PM4 packets | — |

Note: Tegra uses **no copy queue**. Because CPU and GPU share physical LPDDR5 memory, `ctypes.memmove` is just as fast as DMA — there's no PCIe bus to cross.

## Interface Selection

`NVDevice` picks its interface at startup:

```python
# In NVDevice.__init__:
self.iface = self._select_iface(NVKIface, PCIIface, TegraIface)
```

`_select_iface` tries each interface class in order. Each class's `__init__` raises an exception if the hardware doesn't match:

- **NVKIface**: Opens `/dev/nvidiactl` → works on desktop Linux with nvidia.ko
- **PCIIface**: Opens `/dev/vfio/` → works with VFIO passthrough or NVKM
- **TegraIface**: Opens `/dev/nvhost-gpu` → works on Jetson with nvgpu.ko

On your Orin AGX, only `/dev/nvhost-gpu` exists, so **TegraIface wins**. More on TegraIface in [Pill 9](09-jetson-nv-backend-pt1.md).

## The Timeline Protocol in Practice

Let's see how three kernels execute with timeline synchronization:

```python
# Kernel A (matmul)
NVComputeQueue()
    .wait(timeline, 0)           # wait for init
    .exec(matmul_prg, ...)
    .signal(timeline, 1)         # "matmul done"
    .submit(dev)

# Kernel B (relu) — depends on A
NVComputeQueue()
    .wait(timeline, 1)           # wait for matmul
    .exec(relu_prg, ...)
    .signal(timeline, 2)         # "relu done"
    .submit(dev)

# Kernel C (softmax) — depends on B
NVComputeQueue()
    .wait(timeline, 2)           # wait for relu
    .exec(softmax_prg, ...)
    .signal(timeline, 3)         # "softmax done"
    .submit(dev)

# CPU waits for all work
dev.timeline_signal.wait(3)      # block until GPU writes 3
```

Timeline value flow:

```
GPU Signal Memory:  [0] → [1] → [2] → [3]
                     ↑     ↑     ↑     ↑
                    init   A     B     C
                                       │
CPU:  dev.synchronize() ───────────────┘
```

No fences, no events, no stream API. Just an integer that goes up. When the CPU sees `signal.value >= N`, it knows everything up to batch N is done.

### Timeline Wrapping

One edge case: the timeline can overflow after 2³¹ submissions:

```python
def _wrap_timeline_signal(self):
    # Swap to shadow signal, reset counter
    self.timeline_signal, self._shadow_timeline_signal = \
        self._shadow_timeline_signal, self.timeline_signal
    self.timeline_value = 1
    self.timeline_signal.value = 0
```

Two signal buffers ping-pong to avoid ever stalling for a reset.

## Kernel Arguments: The Fast Path

Every kernel launch needs arguments — buffer pointers and scalar values. TinyGrad pre-allocates a big `kernargs_buf` and carves it up with a bump allocator:

```python
# Device init: allocate 16 MB for kernel args
self.kernargs_buf = self.allocator.alloc(16 << 20, BufferSpec(cpu_access=True))
self.kernargs_offset_allocator = BumpAllocator(self.kernargs_buf.size, wrap=True)
```

Each kernel launch bumps the pointer:

```python
def fill_kernargs(self, bufs, vals, kernargs=None):
    argsbuf = kernargs or self.dev.kernargs_buf.offset(
        offset=self.dev.kernargs_offset_allocator.alloc(self.kernargs_alloc_size, 8),
        size=self.kernargs_alloc_size
    )
    return self.args_state_t(argsbuf, self, bufs, vals=vals)
```

The bump allocator wraps around, reusing space from completed kernels. This avoids per-launch allocations entirely — critical for LLM inference where you launch hundreds of tiny kernels per token.

### CLikeArgsState: Packing Layout

```
kernargs buffer layout:
┌──────────┬──────────┬───────┬──────────┬───────┐
│ prefix   │ buf0_ptr │ buf1  │ val0     │ val1  │
│ (opt)    │ (u64)    │ (u64) │ (u32)    │ (u32) │
└──────────┴──────────┴───────┴──────────┴───────┘
     4B ×N     8B ×bufs    4B ×vals
```

Buffer pointers are 64-bit GPU virtual addresses. Values are 32-bit integers (shape dimensions, strides, etc.).

## Profiling

HCQ has built-in profiling via timestamp signals:

```python
with hcq_profile(dev, queue=q, desc="matmul", enabled=PROFILE) as (st, en):
    q.exec(prg, args, global_size, local_size)
# After: en.timestamp - st.timestamp = kernel time in microseconds
```

Enable with:
```bash
PROFILE=1 NV=1 python3 my_script.py
```

This records per-kernel timestamps using GPU hardware timers — far more accurate than Python `time.time()`.

## Summary

- **HCQ** is TinyGrad's vendor-neutral GPU abstraction. 6 classes: Device, Queue, Signal, Program, Buffer, Allocator
- **No CUDA Runtime/Driver** — ioctls directly to the kernel driver
- **Timeline protocol**: monotonically increasing signal values for dependency tracking
- **Interface selection**: NVDevice picks TegraIface on Jetson, NVKIface on desktop
- **Bump allocator** for kernel args avoids per-launch allocation overhead
- **Jetson skips DMA** — unified memory means `memmove` is the copy path
- Build a queue → chain commands → submit → timeline goes up → done

---

**Previous**: [← Pill 4: NVIDIA GPU Deep Dive](04-nvidia-gpu.md)
**Next**: [Pill 6: Schedule & Memory →](06-schedule-memory.md)
