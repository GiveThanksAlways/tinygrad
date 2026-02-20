# Pill 9: Jetson NV Backend, Part 1 — TegraIface

## Why a Whole New Backend?

In [Pill 5](05-device-backends.md), you learned that `NVDevice` picks its interface at startup. On desktop Linux, NVIDIA GPUs use `nvidia.ko` — a proprietary kernel driver with a complex **Resource Manager (RM)** API. Tinygrad's `NVKIface` and `PCIIface` speak this RM language.

On Jetson, the driver is completely different:

```
Desktop (nvidia.ko)              Jetson (nvgpu.ko)
──────────────                   ────────────────
/dev/nvidia0                     /dev/nvgpu/igpu0/ctrl
/dev/nvidia-uvm                  /dev/nvmap
NV_ESC_RM_ALLOC                  NVGPU_GPU_IOCTL_*
NV_ESC_RM_CONTROL                NVGPU_AS_IOCTL_*
RM object hierarchy              Flat ioctl per operation
PCIe BAR for doorbell            MMIO via mmap(ctrl fd)
48-bit VA space                  40-bit VA space
```

Different device files, different ioctl numbers, different memory management, different doorbell mechanism. **TegraIface** is a 766-line translation layer that makes the rest of TinyGrad's NV backend work unchanged on Jetson.

## ioctl Crash Course

An **ioctl** (input/output control) is a Linux system call for device-specific commands. Instead of read/write, you pass a command number and a data struct:

```python
import fcntl
result = fcntl.ioctl(fd, ioctl_number, data_buffer)
```

The ioctl number encodes four things:

```
Bits:  [31:30] [29:16]     [15:8]     [7:0]
       Direction  Size     Type char   Command #

Direction: 0=none, 1=write, 2=read, 3=read+write
Type char: 'G'=GPU, 'N'=nvmap, 'A'=address-space, 'H'=channel, 'T'=TSG
```

TegraIface builds these with helper functions:

```python
def _tegra_IOWR(type_char, nr, sz):
    """Read+Write ioctl: kernel reads our data AND writes results back"""
    return (3 << 30) | (sz << 16) | (ord(type_char) << 8) | nr

# Example: GET_CHARACTERISTICS = IOWR('G', 5, 16)
# = 0xC0104705 — read+write, 16 bytes, GPU type, command 5
_NVGPU_GPU_IOCTL_GET_CHARACTERISTICS = _tegra_IOWR('G', 5, 16)
```

## The Five Device Files

TegraIface talks to five different kernel device files:

| File | What | When |
|------|------|------|
| `/dev/nvgpu/igpu0/ctrl` | GPU control (properties, clocks, setup) | Init |
| `/dev/nvmap` | Memory handle creation & allocation | Alloc |
| Address space fd (from `ALLOC_AS`) | GPU virtual address management | Map |
| TSG fd (from `OPEN_TSG`) | Timeslice group for scheduling | Channel setup |
| Channel fd (from `OPEN_CHANNEL`) | Per-channel command submission | Kernel launch |

Compare desktop: two files (`/dev/nvidia0` + `/dev/nvidiactl`) handle everything through the RM object hierarchy.

## GPU Discovery

When TegraIface starts, the first thing it does is query GPU characteristics:

```python
chars = _nvgpu_gpu_characteristics()
_tegra_ioctl(self._ctrl_fd, _NVGPU_GPU_IOCTL_GET_CHARACTERISTICS, ...)
```

The characteristic struct returns:

```python
chars.arch = 0x170              # ga10b architecture ID
chars.num_gpc = 2               # Graphics Processing Clusters
chars.sm_arch_sm_version = 0x807  # SM 8.7 = Ampere mobile
chars.gpu_va_bit_count = 40     # 40-bit virtual address space (1 TB)
chars.compute_class = 0xC7C0    # AMPERE_COMPUTE_B class number
chars.dma_copy_class = 0xC7B5   # AMPERE_DMA_COPY_B class number
chars.gpfifo_class = 0xC76F     # AMPERE_CHANNEL_GPFIFO_A
```

From this, TinyGrad determines the SM version, PTX target, class numbers for kernel dispatch, and VA space size.

## Memory Allocation: The 5-Step Pipeline

This is the core of TegraIface. Allocating GPU-visible memory is a 5-step process through two kernel subsystems:

```
Step 1: CREATE (nvmap)     → handle (kernel tracking object)
Step 2: ALLOC  (nvmap)     → physical pages committed
Step 3: GET_FD (nvmap)     → dmabuf file descriptor
Step 4: MAP_BUFFER_EX (AS) → GPU virtual address assigned
Step 5: mmap   (libc)      → CPU virtual address mapped
```

### Step 1: Create Handle

```python
create = _nvmap_create_handle(size=size)
_tegra_ioctl(self._nvmap_fd, _NVMAP_IOC_CREATE, create)
handle = create.handle  # just a tracking number — no memory yet
```

### Step 2: Allocate Physical Pages

```python
alloc = _nvmap_alloc_handle(
    handle=handle,
    heap_mask=_NVMAP_HEAP_IOVMM,      # SMMU-managed memory
    flags=(_NVMAP_TAG_TINYGRAD << 16) | cache_flags,
    align=alloc_align,
)
_tegra_ioctl(self._nvmap_fd, _NVMAP_IOC_ALLOC, alloc)
```

Key choices:

- **`_NVMAP_HEAP_IOVMM`** (bit 30): Memory goes through the SMMU (IOMMU). All GPU memory on Orin uses this — the SMMU translates GPU virtual addresses to physical.

- **`_NVMAP_TAG_TINYGRAD`** (0x0900): A tag identifying the allocator. Without it, the kernel logs `nvmap_alloc_handle WARNING: tag not specified` to dmesg.

- **Cache flags**: Two options:
  ```
  WRITE_COMBINE (1): CPU writes coalesced, reads slow. For control structures
  INNER_CACHEABLE (2): Full CPU cache. For compute data. IO-coherent via SMMU
  ```

- **Alignment**: 4KB for small buffers, 2MB for ≥8MB buffers (SMMU TLB efficiency).

### Step 3: Get DMA-buf FD

```python
get_fd = _nvmap_create_handle(handle=handle)
_tegra_ioctl(self._nvmap_fd, _NVMAP_IOC_GET_FD, get_fd)
dmabuf_fd = get_fd.size  # fd reused in the size field (kernel quirk)
```

A **dmabuf** is Linux's abstraction for sharing memory between drivers. The nvmap handle becomes a dmabuf fd that the nvgpu GPU driver can understand.

### Step 4: Map into GPU VA

```python
map_args = _nvgpu_as_map_buffer_ex_args(
    flags=0,             # let kernel pick VA
    compr_kind=-1,       # no compression
    dmabuf_fd=dmabuf_fd,
    page_size=4096,      # 4KB pages on ga10b
)
_tegra_ioctl(self._as_fd, _NVGPU_AS_IOCTL_MAP_BUFFER_EX, map_args)
gpu_va = map_args.offset  # kernel returns assigned GPU VA
```

### Step 5: Map into CPU VA

```python
cpu_addr = libc.mmap(
    gpu_va,              # MAP_FIXED hint: map at same address
    size,
    PROT_READ | PROT_WRITE,
    MAP_SHARED | MAP_FIXED,
    dmabuf_fd,
    0
)
```

**This is the unified memory magic.** By using `MAP_FIXED` at the GPU VA address, the same pointer value works for both CPU and GPU:

```
Address 0x100200000:
  GPU: loads/stores go through GPU MMU → physical page X
  CPU: loads/stores go through CPU MMU → physical page X (same!)
```

`HCQBuffer.va_addr` is simultaneously a valid CPU pointer and GPU address. That's why `ctypes.memmove(va_addr, src, n)` works for copyin, and why the GPU can dereference the same address in kernel arguments.

**Fallback**: If `MAP_FIXED` fails (address collision), we accept whatever address the kernel gives. Then `va_addr` is the GPU VA and the CPU view comes from a separate mmap. TinyGrad handles this transparently through `HCQBuffer.view`.

## The 40-bit VA Problem

Desktop GPUs have 48-bit VA (256 TB of address space). The Orin has **40-bit** (1 TB). And part of that is eaten by hardware:

```
40-bit VA Space (1 TB):
0x00_0000_0000 ┌──────────────────────┐
               │  User allocations     │
               │  (buffers, kernels)   │
               │                       │
0xFD_0000_0000 ├──────────────────────┤
               │  local_mem_window     │  ← GPU hardware intercepts access
               │  (1 GB reserved)      │     for per-thread scratch memory
0xFE_0000_0000 ├──────────────────────┤
               │  shared_mem_window    │  ← GPU hardware intercepts access
               │  (1 GB reserved)      │     for on-chip SRAM (fast)
0xFF_FFFF_FFFF └──────────────────────┘
```

**Memory windows** are special VA ranges. When GPU code does `ld.shared` or `ld.local`, the hardware intercepts addresses in these ranges and redirects them to on-chip SRAM or scratch DRAM — they're not real allocations.

On desktop, windows live at `0x7293_0000_0000` and `0x7294_0000_0000` — safely above any normal allocation. On Jetson, we put them near the top of the 40-bit space:

```python
if self.is_tegra():
    self.shared_mem_window = 0xFE00000000
    self.local_mem_window  = 0xFD00000000
```

### The VA Collision Fix

Without explicit reservation, the kernel's VA allocator could place user buffers at `0xFD...` or `0xFE...`, causing GPU faults. TegraIface reserves these ranges at init:

```python
for window_va in [0xFD00000000, 0xFE00000000]:
    rsv = _nvgpu_as_alloc_space_args(
        pages=0x40000000 // 4096,   # 1GB / 4KB = 262144 pages
        page_size=4096,
        flags=0x1,                   # FIXED_OFFSET
        offset=window_va,
    )
    _tegra_ioctl(self._as_fd, _NVGPU_AS_IOCTL_ALLOC_SPACE, rsv)
```

This "wastes" 2 GB of VA space but prevents random GPU faults during LLM inference, when many large buffer allocations push the VA allocator into the window regions.

## Channel & TSG Setup

### Channels and TSGs

A GPU **channel** is like a CPU thread — it has its own command stream. Each channel has:
- **GPFIFO**: Ring buffer of pointers to command lists (pushbuffers)
- **USERD**: Doorbell register region (CPU writes to wake GPU)
- One or more bound **object contexts** (compute engine, DMA engine)

A **TSG (Timeslice Group)** groups channels for GPU scheduling. TinyGrad creates one TSG with two channels:

```
TSG (Timeslice Group)
├── Compute Channel (kernel launches)
└── DMA Channel (memory copies) — unused on Tegra
```

### The RM Translation

TinyGrad's NVDevice speaks in RM objects (`rm_alloc`, `rm_control`). TegraIface translates each to nvgpu ioctls:

| RM Object | TegraIface Translation |
|-----------|----------------------|
| `NV01_DEVICE_0` | NOP (skip — hierarchy container) |
| `NV01_MEMORY_VIRTUAL` | `ALLOC_AS` + reserve window VAs |
| `KEPLER_CHANNEL_GROUP_A` | `OPEN_TSG` |
| `FERMI_CONTEXT_SHARE_A` | `TSG.CREATE_SUBCONTEXT` |
| `AMPERE_CHANNEL_GPFIFO_A` | Open channel + bind AS + bind TSG + setup |
| `AMPERE_COMPUTE_B` | `CHANNEL.ALLOC_OBJ_CTX` |
| `AMPERE_DMA_COPY_B` | `CHANNEL.ALLOC_OBJ_CTX` |

The most complex translation is `AMPERE_CHANNEL_GPFIFO_A` — it requires **6 ioctls** in sequence:

```python
# 1. Open raw channel
OPEN_CHANNEL → channel_fd

# 2. Bind channel to address space
AS_BIND_CHANNEL(channel_fd)

# 3. Bind channel to TSG (with subcontext)
TSG_BIND_CHANNEL_EX(channel_fd, subcontext_id)

# 4. Disable watchdog (prevent timeout on long kernels)
CHANNEL_WDT(channel_fd, wdt_status=DISABLED)

# 5. Setup with GPFIFO ring + USERD doorbell
CHANNEL_SETUP_BIND(channel_fd, gpfifo_dmabuf, userd_dmabuf, num_entries=1024)

# 6. Enable channel
CHANNEL_ENABLE(channel_fd)
```

### The Small-Buffer Constraint

This was a hard-won lesson. The nvgpu kernel's `SETUP_BIND` handler has strict rules:

1. `gpfifo_dmabuf_offset` **MUST** be 0 (non-zero → `EINVAL`)
2. `userd_dmabuf_offset` **MUST** be 0 (non-zero → `EINVAL`)
3. The kernel maps the **entire** dmabuf into GPU VA

TinyGrad's `NVDevice` normally allocates one big `gpfifo_area` (3 MB on desktop) containing GPFIFO rings and USERD regions for all channels. But passing a 3 MB dmabuf to `SETUP_BIND` would:

- Map 3 MB into GPU VA per channel (redundant overhead)
- Fragment the small 40-bit VA space
- Crash `ALLOC_OBJ_CTX` (GR context fails with fragmented VA)

**Solution**: Allocate small per-channel dmabufs:

```python
gpfifo_ring_size = 1024 * 8   # 8 KB (1024 entries × 8 bytes)
userd_size = 4096              # 4 KB (doorbell region)
```

Then overlay them onto `gpfifo_area` with `MAP_FIXED`:

```python
# Replace the corresponding pages in gpfifo_area's mmap
libc.mmap(gpfifo_area_cpu + offset, ring_size,
          PROT_READ | PROT_WRITE,
          MAP_SHARED | MAP_FIXED,
          gpfifo_ring_dmabuf_fd, 0)
```

This preserves TinyGrad's existing `cpu_view()` API — reads/writes to `gpfifo_area` work identically — but the physical backing is now the small dmabufs that match what `SETUP_BIND` mapped for the GPU.

## The Doorbell

On desktop, the CPU wakes the GPU via a PCIe BAR write — which is slow (microseconds for a round-trip). On Tegra, the doorbell is memory-mapped from the GPU control fd:

```python
def setup_usermode(self):
    addr = libc.mmap(None, 0x10000,
                     PROT_READ | PROT_WRITE,
                     MAP_SHARED,
                     self._ctrl_fd,  # /dev/nvgpu/igpu0/ctrl
                     0)
    return 0, MMIOInterface(addr, 0x10000, fmt='I')  # 32-bit words
```

The doorbell is at offset `0x90` from this base. Writing to it tells the GPU "new work in the GPFIFO ring":

```
CPU writes:
  1. Pushbuffer commands to GPU memory
  2. GPFIFO entry (pointer + length)
  3. Updated GPPUT register
  4. Memory barrier (DSB on ARM)
  5. Doorbell write at offset 0x90  ──→  GPU wakes up

GPU reads:
  1. GPFIFO entry → pushbuffer address
  2. Pushbuffer → GPU commands (QMD, signal, etc.)
  3. Dispatch kernel → execute → write signal
```

The Tegra doorbell is **nanoseconds** (MMIO, shared SoC), not microseconds (PCIe). This speed is actually what causes the QMD race condition — covered in [Pill 10](10-jetson-nv-backend-pt2.md).

## RM Control Translation

Beyond `rm_alloc`, TegraIface also translates `rm_control` calls:

| Control Command | Tegra Translation |
|----------------|-------------------|
| `PERF_BOOST` | Write max freq to `/sys/class/devfreq/17000000.gpu/min_freq` |
| `GPFIFO_SCHEDULE` | NOP (auto-scheduled by nvgpu) |
| `GET_WORK_SUBMIT_TOKEN` | Return saved token from `SETUP_BIND` |
| `GR_GET_INFO` | Derive from characteristics struct |
| `GET_CLASSLIST` | Return known class numbers |
| `GET_GID_INFO` | Synthetic UUID (starts with "J" for Jetson) |
| `FB_FLUSH_GPU_CACHE` | NOP (IO-coherent SMMU handles this) |

The **PERF_BOOST** translation is fun. On desktop, it calls RM to boost GPU clocks. On Tegra:

```python
max_freq = open("/sys/class/devfreq/17000000.gpu/max_freq").read()
open("/sys/class/devfreq/17000000.gpu/min_freq", "w").write(max_freq)
```

This pins the GPU to max frequency — important for benchmarking, since the default governor would downclock during pauses between kernel launches.

## Architecture: Why a Translation Layer?

TegraIface could have been a completely separate code path. Instead, it implements the same `rm_alloc()`/`rm_control()`/`alloc()` interface as `NVKIface`, allowing the rest of `NVDevice` to work unchanged:

```python
class NVDevice(HCQCompiled):
    def __init__(self):
        # This line is the only difference:
        self.iface = self._select_iface(NVKIface, PCIIface, TegraIface)

        # Everything below works identically on desktop and Tegra:
        self.iface.rm_alloc(NV01_DEVICE_0, ...)
        self.iface.rm_alloc(KEPLER_CHANNEL_GROUP_A, ...)
        self.iface.rm_alloc(AMPERE_CHANNEL_GPFIFO_A, ...)
        # ... queue setup, allocator init, compiler init ...
```

From the outside, NVDevice doesn't know or care whether it's talking to nvidia.ko or nvgpu.ko. The translation layer handles all the differences internally.

## Summary

- **TegraIface**: 766-line translation layer — maps RM-style API to nvgpu ioctls
- **5 device files**: ctrl, nvmap, address-space, TSG, channel (vs. 2 on desktop)
- **5-step allocation**: CREATE → ALLOC → GET_FD → MAP_BUFFER_EX → mmap
- **Unified memory magic**: `MAP_FIXED` at GPU VA makes the same address work for both CPU and GPU
- **40-bit VA** with reserved windows at 0xFD/0xFE to prevent collision
- **Small per-channel dmabufs** for GPFIFO rings (nvgpu's SETUP_BIND requires offset=0)
- **Nanosecond doorbell** via MMIO (not PCIe) — faster but causes QMD race

---

**Previous**: [← Pill 8: Pattern Matching & Graph Rewriting](08-pattern-matching.md)
**Next**: [Pill 10: Jetson NV Backend, Part 2 →](10-jetson-nv-backend-pt2.md)
