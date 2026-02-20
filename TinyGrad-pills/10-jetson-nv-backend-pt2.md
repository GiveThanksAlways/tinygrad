# Pill 10: Jetson NV Backend, Part 2 — Runtime Fixes

## The Fixes That Made It Work

[Pill 9](09-jetson-nv-backend-pt1.md) covered TegraIface — the translation layer that lets TinyGrad talk to Jetson's nvgpu driver. But getting ioctls right was only half the battle. The runtime behavior of LLM inference exposed several issues where desktop assumptions broke on Tegra hardware.

This pill covers five runtime fixes, a toolchain fix, and a PTX formatting fix.

## Fix 1: The QMD Race Condition

### Background: How Signals Work

When a GPU kernel finishes, TinyGrad needs to know. The mechanism is **signal release** — the QMD (kernel launch descriptor) tells the GPU "when this kernel is done, write value X to address Y":

```
QMD:
  program_address: 0x1234000
  grid_dim_x: 128
  release0_address: 0xABCD000   ← signal memory location
  release0_payload: 42          ← value to write when done
```

### The Race

On desktop, kernel launch → GPU reads QMD has **microsecond** latency (PCIe round-trip). The CPU is slow enough that by the time it writes the next QMD, the GPU has already read the current one.

On Tegra, the doorbell is **nanosecond** MMIO. The CPU can overwrite a QMD before the GPU reads it:

```
Timeline:
CPU:  [Write QMD A] [Submit A] [Write QMD B (overwrites A)] [Submit B]
GPU:  ................................[Read QMD]........→ reads B's data!
                                      ^^^^^^^^^^^^
                                      GPU hasn't read A yet when CPU
                                      overwrites it with B's payload
```

If QMD A had `release_payload = 42` and QMD B has `release_payload = 43`, the GPU writes 43 when kernel A finishes. The CPU thinks kernel B is done (it's not). Subsequent kernels that depend on B's output read garbage.

**Symptom**: Random incorrect results during LLM inference. Nondeterministic — sometimes works, sometimes doesn't.

### The Fix: `_tegra_signal`

```python
class NVComputeQueue(NVCommandQueue):
    _tegra_signal: ClassVar[bool] = False

    def signal(self, signal, value):
        if self._tegra_signal:
            self.active_qmd = None  # force pushbuffer-based signal
        # ... rest of signal logic
```

Setting `active_qmd = None` forces signals out of the QMD and into the **pushbuffer**:

**Before (QMD-embedded)**:
```
Pushbuffer A:
  COMPUTE_DISPATCH(qmd_addr=0x1000)  ← QMD has signal embedded
  
QMD at 0x1000:
  program=..., grid=..., release_addr=..., release_payload=42
```
Problem: QMD at 0x1000 is shared/reusable memory that gets overwritten.

**After (pushbuffer-based)**:
```
Pushbuffer A (at unique bump-allocated offset 0x0000):
  COMPUTE_DISPATCH(qmd_addr=0x1000)
  MEM_OP_B(address=0xABCD, data=42)   ← signal in pushbuffer

Pushbuffer B (at different offset 0x2000):
  COMPUTE_DISPATCH(qmd_addr=0x1000)
  MEM_OP_B(address=0xABCD, data=43)   ← can't overwrite A's signal
```

Each pushbuffer lives at a unique offset in the command buffer (bump-allocated). Even if the CPU races ahead, A's signal command is immutable — it's at a different memory address than B's.

The flag is set on device init:

```python
# In NVDevice.__init__:
if self.is_tegra():
    NVComputeQueue._tegra_signal = True
```

## Fix 2: Direct memmove (Copyin/Copyout)

### The Desktop Path

On desktop, CPU↔GPU copies go through **DMA staging**:

```
copyin:  CPU buffer → DMA → staging buffer → DMA → GPU buffer
copyout: GPU buffer → DMA → staging buffer → CPU reads staging
```

This makes sense on desktop — GPU VRAM is across PCIe, and DMA is faster than CPU-initiated reads over the bus.

### Why It's Wrong on Tegra

On Jetson, GPU and CPU share the same LPDDR5 RAM. Buffers are allocated with `INNER_CACHEABLE` — the CPU cache is valid for these addresses. The DMA staging path is strictly worse:

- 2× data movement (original + staging copy)
- Per-chunk submission overhead
- Read-back from uncached staging buffer

### The Fix

```python
class NVAllocator(HCQAllocator['NVDevice']):
    def _copyout(self, dest: memoryview, src: HCQBuffer):
        if self.dev.is_tegra():
            self.dev.synchronize()  # wait for GPU to finish writing
            ctypes.memmove(from_mv(dest), int(src.va_addr), len(dest))
            return
        super()._copyout(dest, src)

    def _copyin(self, dest: HCQBuffer, src: memoryview):
        if self.dev.is_tegra():
            self.dev.synchronize()  # wait for GPU to finish reading
            ctypes.memmove(int(dest.va_addr), from_mv(src), len(src))
            return
        super()._copyin(dest, src)
```

Why `synchronize()` first?
- **copyout**: The GPU might still be writing to `src`. We must wait until the kernel finishes.
- **copyin**: The GPU might be reading from `dest` in an in-flight kernel. We must wait until it's done.

Why `va_addr` works as a CPU pointer? Because of the `MAP_FIXED` trick from [Pill 9](09-jetson-nv-backend-pt1.md) — the same address is valid for both CPU and GPU.

## Fix 3: Shader Local Memory Synchronization

### The Bug

When a GPU kernel needs more registers than hardware provides, the compiler "spills" to **local memory** — per-thread scratch space in DRAM. TinyGrad tracks this with `shader_local_mem`:

```python
self.slm_per_thread = 0
self.shader_local_mem = None
```

When a new kernel needs more, it reallocates:

```python
if new_kernel.local_mem_per_thread > self.slm_per_thread:
    # Need bigger buffer!
    self.shader_local_mem = realloc(bigger_size)  # frees old
```

But what if a previous kernel is **still running** on the GPU, using the old buffer?

```
CPU: [launch kernel A (32B/thread)] [realloc local mem] [launch kernel B (64B/thread)]
GPU: .....................[kernel A still running, using old buffer]...→ freed memory! → CRASH
```

### The Fix

```python
def _ensure_has_local_memory(self, required):
    if self.slm_per_thread >= required:
        return
    self.synchronize()          # ← wait for all GPU work to finish
    self.slm_per_thread = round_up(required, 32)
    # ... realloc ...
```

One line: `self.synchronize()`. This only triggers during the first few kernel launches when TinyGrad discovers the maximum local memory requirement. After warmup, it's a no-op (the buffer is big enough and never reallocated again).

## Fix 4: PMA Disabled on Tegra

```python
self.pma_enabled = PMA.value > 0 and PROFILE >= 1 and not self.is_tegra()
```

**PMA** (Performance Monitoring Aggregator) is NVIDIA's hardware performance counter system. On desktop, it works through the `GT200_DEBUGGER` RM class. On Tegra, `nvgpu.ko` doesn't expose debugger objects — attempting PMA would crash. Simple fix: disable it.

## Fix 5: Fault Reporting

When the GPU hangs, TinyGrad tries to read detailed fault info from the debugger. On Tegra this isn't available:

```python
if self.is_tegra():
    raise RuntimeError(
        "GPU hang detected on Tegra device "
        "(no detailed fault info available via nvgpu)"
    )
```

Clean error instead of a secondary crash. GPU hangs on Tegra are debugged via `dmesg` (nvgpu.ko logs detailed fault info there).

## Fix 6: BEAM fork vs spawn

### The Problem

BEAM search ([Pill 7](07-beam-search.md)) uses multiprocessing to compile kernel candidates in parallel. Python's multiprocessing has two modes:

- **fork**: Copy parent process memory (fast, shares file descriptors)
- **spawn**: Start fresh process (slow, clean state)

On Linux, BEAM defaulted to `fork`. But there's a subtle problem when the NV device has been initialized:

```python
# Parent process:
device = NVDevice()  # opens /dev/nvhost-gpu, mmaps memory, etc.

# fork() → child process inherits file descriptors
# Child tries to compile a kernel:
compiler.compile(ptx_source)  # uses NVRTC which may touch inherited fds
# → Segfault or hanging child process
```

The forked child inherits the parent's GPU file descriptors and memory mappings. If the child (or NVRTC inside it) touches these, it can corrupt the parent's GPU state.

### The Fix

BEAM workers are initialized with device access disabled:

```python
def _init_worker():
    Context(ALLOW_DEVICE_USAGE=0, VIZ=0, TRACK_MATCH_STATS=0).__enter__()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
```

And the multiprocessing context is `fork` on Linux (for speed), since the workers only compile — they never touch GPU devices:

```python
_mp_ctx = "fork" if sys.platform == "linux" else "spawn"
beam_pool = multiprocessing.get_context(_mp_ctx).Pool(workers, _init_worker)
```

The key insight: compilation (PTX → CUBIN via NVRTC) doesn't need GPU access. Only timing does — and timing happens in the parent process.

## Fix 7: PTX Vector Load Formatting

### The Bug

PTX vector loads have a specific syntax:

```ptx
// Correct: register list in braces
ld.global.v4.f32 {%f0, %f1, %f2, %f3}, [%rd0];

// Wrong: some formatting edge case produced
ld.global.v4.f32 %f0, %f1, %f2, %f3, [%rd0];
```

The PTX renderer had a formatting issue with vector loads/stores on certain UOp patterns. `ptxas` (the PTX assembler) rejected the malformed instruction.

### The Fix

The PTXRenderer was patched to correctly format vector operands with braces for all vector width combinations. This was a one-line fix in the renderer output formatting.

## Fix 8: CUDA_INCLUDE_PATH for NixOS

### The Problem

NVRTC needs CUDA headers to compile PTX. On standard Linux:
```
/usr/local/cuda/include/cuda.h
```

On NixOS, nothing lives at standard paths. CUDA is in the Nix store:
```
/nix/store/abc123-cuda-cudart-12.x/include/cuda.h
```

### The Fix

```python
CUDA_INCLUDE_PATH = getenv("CUDA_INCLUDE_PATH", "")
if CUDA_INCLUDE_PATH:
    self.compile_options += [f"-I{CUDA_INCLUDE_PATH}"]
```

Set in the Nix dev shell:
```nix
shellHook = ''
  export CUDA_INCLUDE_PATH="${cudaPackages.cuda_cudart}/include"
'';
```

## All Fixes Summary

| Fix | Problem | Root Cause | Solution |
|-----|---------|-----------|----------|
| QMD Race | Wrong signal values | MMIO doorbell too fast | Pushbuffer-based signals |
| Direct memmove | Slow copy on unified mem | DMA staging unnecessary | `ctypes.memmove` directly |
| Local mem sync | Freed buffer while in use | No wait before realloc | `synchronize()` before free |
| PMA disabled | Crash on profiling | No debugger on nvgpu | Skip PMA on Tegra |
| Fault reporting | Secondary crash on hang | No RM debugger | Clean error message |
| BEAM fork | Child process corruption | Inherited GPU fds | `ALLOW_DEVICE_USAGE=0` |
| PTX vectors | Compile error | Formatting bug | Fix brace syntax |
| CUDA path | Missing headers on NixOS | Non-standard paths | `CUDA_INCLUDE_PATH` env |

## How to Test These Fixes

```bash
# Quick smoke test: run a simple computation
NV=1 python3 -c "from tinygrad import Tensor; print(Tensor.randn(8).realize().numpy())"

# Test with BEAM (exercises fork fix + signal fix under load)
BEAM=2 NV=1 python3 -c "
from tinygrad import Tensor
a = Tensor.randn(256,256)
print((a @ a.T).realize().numpy()[:2,:2])
"

# Test copyin/copyout
NV=1 python3 -c "
from tinygrad import Tensor
import numpy as np
x = np.random.randn(1000).astype(np.float32)
t = Tensor(x)
t.realize()
assert np.allclose(t.numpy(), x), 'copyin/copyout failed'
print('copyin/copyout OK')
"
```

---

**Previous**: [← Pill 9: Jetson NV Backend, Part 1](09-jetson-nv-backend-pt1.md)
**Next**: [Pill 11: The Matvec Heuristic & 7.6× Speedup →](11-matvec-heuristic.md)
