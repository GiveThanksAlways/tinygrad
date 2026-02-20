# Pill 4: NVIDIA GPU Deep Dive — CUDA, PTX, and the Orin

## Three Layers of NVIDIA

NVIDIA GPUs have three software layers you need to understand:

```
Your Code (Python/C)
    │
    ▼
CUDA Runtime (libcudart.so)    ← High-level: malloc, memcpy, launch
    │
    ▼
CUDA Driver (libcuda.so)       ← Low-level: contexts, modules, streams
    │
    ▼
Kernel Driver (nvidia.ko)      ← Kernel: ioctls, memory management, hardware
    │                              On Jetson: nvgpu.ko instead
    ▼
GPU Hardware (SM 8.7)          ← Silicon: warps, SMs, memory controllers
```

**TinyGrad's NV backend bypasses CUDA Runtime and Driver entirely.** It talks directly to the kernel driver via ioctls. More in [Pill 5](05-device-backends.md).

## PTX: The GPU Assembly Language

### What is PTX?

**Parallel Thread Execution** (PTX) is NVIDIA's virtual ISA — an assembly language for their GPUs. It's "virtual" because the hardware doesn't execute PTX directly. PTX gets compiled to **SASS** (the real hardware instructions) by `ptxas` or NVRTC.

```
Python Tensor Ops
    │ (tinygrad codegen)
    ▼
PTX Assembly          ← human-readable, portable across GPU generations
    │ (nvrtc / ptxas)
    ▼
SASS Binary (CUBIN)   ← actual hardware instructions, GPU-specific
    │ (loaded into GPU memory)
    ▼
Hardware Execution
```

### Why PTX Instead of SASS?

- **Portable**: PTX for SM 7.0 works on SM 8.7 (recompiled automatically)
- **Readable**: You can debug it. SASS is opaque binary
- **Stable**: PTX ISA changes slowly. SASS changes every generation
- **What TinyGrad uses**: The PTXRenderer emits PTX, then NVRTC compiles to CUBIN

### PTX Basics

PTX looks like typed RISC assembly:

```ptx
.version 8.0
.target sm_87          // Orin's SM version
.address_size 64

.visible .entry my_kernel(
    .param .u64 buf_in,
    .param .u64 buf_out
) {
    .reg .f32 %f<4>;     // 4 float registers
    .reg .u32 %r<4>;     // 4 uint registers
    .reg .u64 %rd<4>;    // 4 u64 registers

    // Get thread index
    mov.u32 %r0, %tid.x;

    // Calculate address: buf_in + tid * 4
    mul.wide.u32 %rd0, %r0, 4;
    ld.param.u64 %rd1, [buf_in];
    add.u64 %rd2, %rd1, %rd0;

    // Load value
    ld.global.f32 %f0, [%rd2];

    // Compute: val * 2.0 + 1.0
    mul.f32 %f1, %f0, 0f40000000;     // 2.0 in IEEE 754 hex
    add.f32 %f2, %f1, 0f3f800000;     // 1.0 in IEEE 754 hex

    // Store result
    ld.param.u64 %rd3, [buf_out];
    add.u64 %rd3, %rd3, %rd0;
    st.global.f32 [%rd3], %f2;

    ret;
}
```

### Key PTX Concepts

**Registers**: Virtual (unlimited). `ptxas` maps them to physical registers.
```ptx
.reg .f32 %f<100>;    // 100 float registers
.reg .pred %p<10>;    // 10 predicate (boolean) registers
```

**Memory spaces**:
```ptx
ld.global.f32    // Global memory (DRAM) — slow, large
ld.shared.f32    // Shared memory (SRAM) — fast, small, per-block
ld.local.f32     // Local memory (DRAM) — register spill space
ld.const.f32     // Constant memory — cached, read-only
ld.param.u64     // Kernel parameters — in constant memory
```

**Thread indexing**:
```ptx
%tid.x       // threadIdx.x (within block)
%ntid.x      // blockDim.x
%ctaid.x     // blockIdx.x (within grid)
%nctaid.x    // gridDim.x
```

**Predicates** (conditional execution without branches):
```ptx
setp.lt.f32 %p0, %f0, 0f00000000;    // p0 = (f0 < 0.0)
@%p0 mov.f32 %f0, 0f00000000;        // if (p0) f0 = 0.0  (ReLU!)
```

This is how TinyGrad implements `relu()` without branch divergence.

## TinyGrad's PTX Renderer

The `PTXRenderer` class in `renderer/ptx.py` converts UOp lists into PTX source code. Each UOp maps to one or a few PTX instructions:

| UOp | PTX |
|-----|-----|
| `Ops.ADD` | `add.f32 %f2, %f0, %f1` |
| `Ops.MUL` | `mul.f32 %f2, %f0, %f1` |
| `Ops.LOAD` | `ld.global.f32 %f0, [%rd0]` |
| `Ops.STORE` | `st.global.f32 [%rd0], %f0` |
| `Ops.MAX` | `max.f32 %f2, %f0, %f1` |
| `Ops.CAST` (fp16→fp32) | `cvt.f32.f16 %f0, %h0` |
| `Ops.RANGE` | `mov.u32; loop:; setp; @!p bra loop` |
| `Ops.SPECIAL` (tid) | `mov.u32 %r0, %tid.x` |
| `Ops.BARRIER` | `bar.sync 0` |

### Vector Loads

Loading 4 floats at once (128 bits) is more efficient than 4 separate loads:

```ptx
// Scalar: 4 transactions
ld.global.f32 %f0, [%rd0];
ld.global.f32 %f1, [%rd0+4];
ld.global.f32 %f2, [%rd0+8];
ld.global.f32 %f3, [%rd0+12];

// Vector: 1 transaction
ld.global.v4.f32 {%f0, %f1, %f2, %f3}, [%rd0];
```

TinyGrad's `UPCAST` optimization often enables this — processing 4 elements per thread means loading 4 at once.

## Compute Capability & SM Version

Each NVIDIA GPU generation has a **compute capability** (CC) that determines its feature set:

| CC | Architecture | Example GPUs | Features |
|----|-------------|-------------|----------|
| 7.0 | Volta | V100 | Tensor cores (FP16) |
| 7.5 | Turing | RTX 2080 | INT8 tensor cores |
| 8.0 | Ampere | A100 | TF32, BF16, sparsity |
| 8.6 | Ampere | RTX 3090 | Same as 8.0, desktop |
| **8.7** | **Ampere** | **Orin ga10b** | Mobile Ampere |
| 8.9 | Ada Lovelace | RTX 4090 | FP8 tensor cores |
| 9.0 | Hopper | H100 | Transformer engine |

### SM 8.7: The Orin's GPU

The ga10b in the Orin is SM 8.7 — a mobile variant of Ampere. Compared to desktop Ampere (8.0/8.6):

| Feature | A100 (8.0) | RTX 3090 (8.6) | Orin ga10b (8.7) |
|---------|-----------|----------------|-----------------|
| SMs | 108 | 82 | 16 |
| CUDA cores | 6912 | 10496 | 2048 |
| Memory | 80 GB HBM2e | 24 GB GDDR6X | 64 GB LPDDR5 (shared) |
| Bandwidth | 2 TB/s | 936 GB/s | ~102 GB/s (effective) |
| Tensor Cores | Yes | Yes | **Yes** (FP16, INT8) |
| TDP | 400W | 350W | 60W (whole SoC) |

**Key differences for TinyGrad**:
- **16 SMs** means less parallelism → need higher per-SM utilization
- **102 GB/s** bandwidth → LLM decode speed is fundamentally limited
- **Unified memory** → no PCIe transfers, but bandwidth is lower than HBM
- **SM 8.7 ISA** is compatible with SM 8.0 PTX → same codegen works

### How TinyGrad Detects the SM Version

```python
# In ops_nv.py, NVDevice.__init__:
self.sm_version = ...  # 0x807 for SM 8.7

# Decoded:
# (0x807 >> 8) & 0xFF = 8  → major version
# 0x807 & 0xFF = 7         → minor version (but uses lower nibble: 0x7)

self.arch = f"sm_{(self.sm_version>>8)&0xff}{val}" # "sm_87"
```

This `arch` string gets passed to PTXRenderer to emit the correct `.target sm_87` directive.

## CUBIN: The Binary Format

NVRTC compiles PTX into a **CUBIN** — a GPU binary in ELF format. Inside:

```
CUBIN (ELF binary)
├── .text.my_kernel       — SASS instructions (the actual GPU machine code)
├── .nv.info              — Register count, stack size, etc.
├── .nv.info.my_kernel    — Per-kernel metadata
├── .nv.shared.my_kernel  — Shared memory size
├── .nv.constant0         — Constant buffer data
└── Relocation entries     — Address fixups for loading at any GPU VA
```

TinyGrad parses these ELF sections in `NVProgram.__init__()`:

```python
image, sections, relocs = elf_loader(self.lib, force_section_align=128)

for sh in sections:
    if sh.name == f".nv.shared.{self.name}":
        self.shmem_usage = round_up(0x400 + sh.header.sh_size, 128)
    if sh.name == f".text.{self.name}":
        prog_addr = self.lib_gpu.va_addr + sh.header.sh_addr
    elif m := re.match(r'\.nv\.constant(\d+)', sh.name):
        self.constbufs[int(m.group(1))] = (addr, size)
    elif sh.name.startswith(".nv.info"):
        # Parse: register count, local memory, etc.
```

This metadata determines `max_threads` (how many threads can run based on register usage), shared memory configuration, and local memory allocation.

## QMD: Queue Meta Data

The **QMD** is the control structure that describes a kernel launch. Think of it as the GPU equivalent of `exec(program, args)`:

```
QMD (Queue Meta Data) — 64 or 96 DWORDs
├── program_address         — where the SASS code lives in GPU memory
├── grid_dim_x/y/z          — how many blocks (blockIdx)
├── block_dim_x/y/z         — how many threads per block
├── register_count          — registers per thread
├── shared_memory_size      — shared memory per block
├── constant_buffer_addr[]  — pointers to kernel arguments
├── release_address+payload — signal to write when kernel finishes
└── ...
```

TinyGrad builds the QMD in `NVProgram.__init__()` and fills in per-launch values in `NVComputeQueue.exec()`:

```python
# Build template QMD once per program
self.qmd = QMD(dev,
    program_address_upper=hi32(prog_addr),
    program_address_lower=lo32(prog_addr),
    register_count=self.regs_usage,
    shared_memory_size=self.shmem_usage, ...)

# Per-launch: fill grid/block dims + kernel args
qmd.write(cta_raster_width=global_size[0], ...)
qmd.set_constant_buf_addr(0, args_buf.va_addr)
```

## Compilation Options on the Orin

TinyGrad supports three compilers for NVIDIA:

| Compiler | Backend | Speed | Size | Notes |
|----------|---------|-------|------|-------|
| `NVRTCCompiler` | nvrtc (PTX→CUBIN) | Fast | Medium | Default for NV=1 |
| `NVCCCompiler` | nvcc (CUDA→CUBIN) | Slower | Smaller | Needs nvcc installed |
| `NVPTXCompiler` | ptxas (PTX→CUBIN) | Fast | Smaller | Lower-level |
| `NAKRenderer` | Mesa (NIR→CUBIN) | Variable | Variable | Open-source, no NVIDIA libs |

On the Orin with NV=1, the default flow is:

```
UOps → PTXRenderer → PTX source → NVRTCCompiler → CUBIN → NVProgram → GPU
```

### NixOS CUDA Path Fix

On NixOS, CUDA headers aren't at `/usr/local/cuda/include` — they're in the Nix store. Our fix:

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

## Tensor Cores

Tensor cores are specialized hardware units for matrix multiply-accumulate:

```
D[M,N] = A[M,K] × B[K,N] + C[M,N]
```

In one operation, a tensor core computes a small matrix multiply across a warp.

### Tensor Cores on the Orin (SM 8.7)

| Precision | Matrix Size | Throughput |
|-----------|------------|------------|
| FP16 × FP16 → FP32 | 16×16×16 | Highest |
| INT8 × INT8 → INT32 | 16×16×16 | High |
| TF32 × TF32 → FP32 | 16×8×8 | Medium |

TinyGrad's `OptOps.TC` applies tensor core optimizations:

```python
# In heuristic.py
if USE_TC > 0:
    tk.apply_opt(Opt(OptOps.TC, 0, (TC_SELECT, TC_OPT, USE_TC)))
```

This restructures the UOp graph so that matmul inner loops emit WMMA (Warp Matrix Multiply-Accumulate) instructions instead of scalar FMA loops.

**When tensor cores help**: Large matmuls during prefill (prompt processing). Not during decode (matvec) — tensor cores need matrix × matrix.

## Debugging GPU Code

### DEBUG Levels

```bash
DEBUG=1  # Timing info
DEBUG=2  # Detailed timing + kernel shapes
DEBUG=3  # Show fused kernels + optimization choices
DEBUG=4  # Show generated PTX/CUDA source code
DEBUG=5  # Show pre-optimization AST
DEBUG=7  # Show SASS disassembly (the actual hardware instructions)
```

### Example: DEBUG=4 Output

```
$ DEBUG=4 NV=1 python3 -c "from tinygrad import Tensor; (Tensor.randn(4)*2).realize()"

.version 8.0
.target sm_87
.address_size 64
.visible .entry E_4(.param .u64 buf0, .param .u64 buf1) {
  .reg .f32 %f<3>;
  .reg .u64 %rd<3>;
  mov.u32 %r0, %tid.x;
  mul.wide.u32 %rd0, %r0, 4;
  ld.param.u64 %rd1, [buf0];
  add.u64 %rd1, %rd1, %rd0;
  ld.global.f32 %f0, [%rd1];
  mul.f32 %f1, %f0, 0f40000000;
  ld.param.u64 %rd2, [buf1];
  add.u64 %rd2, %rd2, %rd0;
  st.global.f32 [%rd2], %f1;
  ret;
}
```

You can trace every line back to the original tensor operation.

## Summary

- **PTX** is NVIDIA's portable assembly. TinyGrad generates PTX, NVRTC compiles to CUBIN
- **SM 8.7** (Orin) is mobile Ampere — same ISA as A100 but fewer SMs and lower bandwidth
- **QMD** describes each kernel launch (program address, grid size, shared memory, etc.)
- **Tensor cores** accelerate matrix multiply but don't help with matvec (LLM decode)
- **CUBIN** is ELF format — TinyGrad parses it to extract register/memory metadata
- **Debug with DEBUG=4** to see the actual PTX that runs on GPU

---

**Previous**: [← Pill 3: GPU Architecture Primer](03-gpu-architecture.md)
**Next**: [Pill 5: Device Backends & HCQ →](05-device-backends.md)
