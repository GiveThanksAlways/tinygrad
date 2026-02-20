# Pill 3: GPU Architecture Primer

## Why This Matters

TinyGrad generates GPU kernels. To understand *why* it makes certain optimization decisions (LOCAL=4, UPCAST=2, GROUP=32), you need a mental model of how GPUs actually execute code. This pill gives you that model.

**Assumed knowledge**: You know what a CPU is, what threads are, and what memory is. You have a physics/CS background.

## CPU vs GPU: The Core Trade-off

**CPU**: Few cores (4-16), each very fast. Optimized for **latency** — doing one thing quickly.

**GPU**: Many cores (128-16384), each modest speed. Optimized for **throughput** — doing many things at once.

```
CPU (12 cores × 4 GHz):         GPU (2048 cores × 1.3 GHz):
┌──────────────────────┐        ┌──────────────────────────┐
│ Core0 [████████████] │        │ ░░░░░░░░░░░░░░░░░░░░░░░░ │
│ Core1 [████████████] │        │ ░░░░░░░░░░░░░░░░░░░░░░░░ │
│ ...                  │        │ ░░░░░░░░░░░░░░░░░░░░░░░░ │
│ Core11[████████████] │        │ ... (2048 simple cores)   │
│                      │        │ ░░░░░░░░░░░░░░░░░░░░░░░░ │
│ Big caches per core  │        │ Tiny caches, huge BW      │
│ Branch prediction    │        │ No branch prediction      │
│ Out-of-order exec    │        │ In-order, SIMT            │
└──────────────────────┘        └──────────────────────────┘
```

**Why GPUs for ML?** Neural network operations (matmul, convolution, attention) are massively parallel — same operation on millions of independent data elements. Perfect GPU workload.

## GPU Hierarchy: From Grid to Thread

GPU execution has a strict hierarchy. Let's go top-down:

### Grid → Blocks → Warps → Threads

```
Grid (the entire kernel launch)
├── Block 0 (runs on SM 0)
│   ├── Warp 0 (32 threads, lockstep)
│   │   ├── Thread 0
│   │   ├── Thread 1
│   │   └── ... Thread 31
│   ├── Warp 1
│   └── ... more warps
├── Block 1 (runs on SM 1)
│   └── ...
└── Block N (runs on SM N)
```

### Thread

The smallest unit of execution. Has its own:
- **Registers** (fast, private, ~255 per thread on NVIDIA)
- **Program counter**
- **Thread ID** (`threadIdx.x`, `.y`, `.z`)

### Warp (32 threads)

32 threads that execute the **same instruction at the same time** (SIMT — Single Instruction Multiple Threads). This is the fundamental execution unit.

**Key insight**: If threads in a warp take different branches (`if/else`), both branches execute — threads not on the active branch are masked off. This is **warp divergence** and it halves performance.

```
// Good: all threads take same path
if (threadIdx.x < 1024) { ... }  // All 32 threads: TRUE

// Bad: warp divergence
if (threadIdx.x % 2 == 0) { ... }  // 16 threads: TRUE, 16: FALSE
else { ... }                         // Both branches execute!
```

### Block (Thread Block / Workgroup)

A group of warps that:
- Run on the **same SM** (Streaming Multiprocessor)
- Share **shared memory** (fast, on-chip, 48-164 KB)
- Can **synchronize** with barriers (`__syncthreads()`)
- Have a **block ID** (`blockIdx.x`, `.y`, `.z`)

Max threads per block: **1024** (NVIDIA)

### Grid

The entire collection of blocks. Blocks are distributed across SMs by the GPU scheduler. Blocks are independent — no synchronization between blocks (during a single kernel launch).

## Memory Hierarchy

This is what determines GPU performance. From fastest to slowest:

```
Speed            Memory          Size           Scope
─────            ──────          ────           ─────
~1 cycle         Registers       ~256 KB/SM     Thread
~5 cycles        Shared Memory   48-164 KB/SM   Block
~200 cycles      L2 Cache        4-6 MB         GPU
~400 cycles      DRAM (VRAM)     8-80 GB        GPU
~1000 cycles     System RAM      16-256 GB      CPU+GPU (Tegra)
```

### Registers

Fastest. Each thread gets up to ~255 32-bit registers. The compiler maps variables to registers. If a kernel needs too many → **register spilling** to local memory (slow, actually DRAM).

**Why this matters in TinyGrad**: The number of registers per thread limits occupancy (how many warps can fit on an SM). `UPCAST=4` means each thread processes 4 elements — more registers needed — fewer warps — but better data reuse. Classic trade-off.

### Shared Memory (SMEM)

On-chip SRAM shared by all threads in a block. Fast (5 cycles), but small (48KB default, up to 164KB configurable on modern GPUs).

**Uses in TinyGrad**:
- **GROUP reduction**: Threads do partial sums, then combine via shared memory
- **Matvec**: Cooperating threads share loaded weights through SMEM
- **Tile loading**: Load a tile of data into SMEM, then compute from SMEM

### Global Memory (DRAM)

The big one. This is where tensors live. Slow (~400 cycles) but large. The key to performance is minimizing global memory accesses.

**Coalescing**: When adjacent threads access adjacent memory addresses, the GPU combines them into one transaction. Accessing `buf[threadIdx.x]` is coalesced (fast). Accessing `buf[threadIdx.x * stride]` with large stride is uncoalesced (slow).

```
Coalesced (good):           Uncoalesced (bad):
Thread 0 → buf[0]          Thread 0 → buf[0]
Thread 1 → buf[1]          Thread 1 → buf[128]
Thread 2 → buf[2]          Thread 2 → buf[256]
Thread 3 → buf[3]          Thread 3 → buf[384]
= 1 memory transaction     = 4 memory transactions
```

### L2 Cache

Between shared memory and DRAM. Automatic. On the Jetson Orin, L2 is 4MB. Helps when multiple blocks access the same data.

## Streaming Multiprocessors (SMs)

An SM is the GPU's compute unit — like a CPU core but simpler and with more hardware threads.

### What's Inside an SM?

```
SM (Streaming Multiprocessor)
├── Warp Schedulers (4)      — pick which warps to run
├── FP32 CUDA Cores (128)   — basic math (add, mul)
├── FP16 Tensor Cores (if available) — matrix multiply
├── Load/Store units (32)    — memory access
├── Special Function Units (4) — sin, cos, exp, sqrt
├── Register File (256 KB)   — thread registers
└── Shared Memory (48-164 KB) — block-shared fast SRAM
```

### SM on the Jetson Orin (ga10b, SM 8.7)

The Orin's GPU has:
- **16 SMs** (8 GPCs × 2 TPCs × 1 SM/TPC)
- **2048 CUDA cores** (128 per SM × 16 SMs)
- **48 Warps max per SM** (= 1536 threads per SM)
- **256 KB registers per SM**
- **48-164 KB shared memory per SM** (configurable)
- **SM 8.7** = Ampere architecture, minor variant 7

## Occupancy

**Occupancy** = active warps / max warps per SM. Higher occupancy = better latency hiding (while one warp waits for memory, another runs).

Occupancy is limited by:
1. **Registers per thread**: More registers → fewer warps fit
2. **Shared memory per block**: More SMEM → fewer blocks fit
3. **Threads per block**: Max 1024

**Example** (SM 8.7):
- Max 48 warps/SM, 256KB registers
- Kernel uses 64 registers/thread:
  - 64 regs × 32 threads/warp = 2048 regs/warp
  - 256KB / 4B per reg = 65536 regs total
  - 65536 / 2048 = 32 warps max → 32/48 = 67% occupancy

## Memory Bandwidth: The Real Bottleneck

For ML inference (especially LLM decode), the bottleneck isn't compute — it's **memory bandwidth**. Every generated token reads the entire model from memory.

### Orin Bandwidth

```
LPDDR5 theoretical:  ~204 GB/s
LPDDR5 practical:    ~102 GB/s (50% efficiency is typical)
```

### Roofline Model

The **roofline model** tells you whether a kernel is compute-bound or memory-bound:

```
Operational Intensity = FLOPs / Bytes loaded

If OI < peak_FLOPS / peak_BW → memory-bound
If OI > peak_FLOPS / peak_BW → compute-bound
```

For the Orin:
- Peak FP32: ~5.3 TFLOPS
- Peak BW: ~102 GB/s (practical)
- Crossover: 5300 / 102 ≈ 52 FLOPS/byte

**Matvec** (LLM decode): ~2 FLOPS/byte (multiply-add per weight byte). **Extremely** memory-bound. This is why memory bandwidth optimization matters 100× more than compute optimization for LLM inference.

**Matmul** (LLM prefill): ~200+ FLOPS/byte with tiling. Compute-bound. This is where tensor cores shine.

## How TinyGrad Maps to GPU Hardware

| TinyGrad Concept | GPU Hardware |
|------------------|--------------|
| `GLOBAL` axis | `blockIdx` (grid dimension) |
| `LOCAL` axis | `threadIdx` (block dimension) |
| `GROUP` reduction | Shared memory + `__syncthreads` |
| `UPCAST` | Multiple values per thread (registers) |
| `UNROLL` | Compiler unrolls loop body |
| Buffer | Global memory allocation (DRAM) |
| `graph_rewrite` | GPU code that runs on SMs |

### Concrete Example

```python
# TinyGrad: y = x.sum()  where x is (1024,)
# After heuristic: GROUP=32, LOCAL=4

# GPU execution:
# Grid: 1 block
# Block: 4 threads (LOCAL=4)
# Each thread reduces 256 elements (1024/4)
# GROUP=32: Each thread handles 32-element chunks, shared-mem combine

# Thread 0: sum(x[0:256])   → partial_sum[0]
# Thread 1: sum(x[256:512]) → partial_sum[1]
# Thread 2: sum(x[512:768]) → partial_sum[2]
# Thread 3: sum(x[768:1024])→ partial_sum[3]
# __syncthreads()
# Thread 0: total = partial_sum[0] + ... + partial_sum[3]
# Store total → y[0]
```

## Key GPU Programming Pitfalls

### 1. Warp Divergence
Different control flow paths within a warp → serialized execution. TinyGrad avoids this by design — most operations are element-wise.

### 2. Uncoalesced Memory Access
Non-sequential access patterns waste bandwidth. TinyGrad's scheduler tries to arrange memory access so adjacent threads read adjacent addresses.

### 3. Shared Memory Bank Conflicts
Shared memory has 32 banks. If multiple threads in a warp access the same bank, accesses serialize. TinyGrad doesn't manually manage bank conflicts (yet).

### 4. Register Pressure
Too many registers → register spilling → local memory (DRAM speed). UPCAST increases register usage. BEAM search finds the sweet spot.

### 5. Low Occupancy
Too few active warps → can't hide memory latency. But sometimes low occupancy with high per-thread work (UPCAST) wins. Depends on the kernel.

## Putting It Together

When TinyGrad decides `GLOBAL=128, LOCAL=4, UPCAST=2, GROUP=32` for a reduction kernel, it's saying:

- **128 blocks** distributed across 16 SMs (~8 blocks/SM)
- **4 threads per block** = 4/32 of a warp (partial warp, not great)
- Each thread computes **2 output elements** (UPCAST)
- Reduction split across **32 cooperating threads** via shared memory

BEAM search might find `GLOBAL=64, LOCAL=8, UPCAST=4, GROUP=16` does better — different occupancy/register/shared-memory trade-off.

There is no universal optimal. It depends on the kernel, the sizes, and the hardware. That's why BEAM exists.

## Summary

- GPUs trade per-thread speed for massive parallelism
- **Warp** (32 threads) is the fundamental execution unit
- Memory hierarchy: registers → shared → L2 → DRAM
- Memory bandwidth is THE bottleneck for LLM decode
- TinyGrad's axis types (GLOBAL/LOCAL/UPCAST/GROUP) map directly to GPU hardware concepts
- BEAM search explores the hardware trade-off space

---

**Previous**: [← Pill 2: The Compilation Pipeline](02-compilation-pipeline.md)
**Next**: [Pill 4: NVIDIA GPU Deep Dive →](04-nvidia-gpu.md)
