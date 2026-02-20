# Pill 12: Benchmarking & Performance

## The Numbers

Here's what we measured on the Jetson AGX Orin 64GB running Qwen3 1B Q6_K:

| Framework | Backend | tok/s | Notes |
|-----------|---------|-------|-------|
| **tinygrad** (no BEAM) | NV=1 | 7.59 | Heuristic only, first run |
| **tinygrad** (BEAM cached) | NV=1 | **36.71** | After BEAM cache warm |
| **llama.cpp** | CUDA | 25.59–27.80 | Well-optimized C++ |
| **vLLM** | — | N/A | Failed to install on Orin |
| **MLC-LLM** | — | N/A | Build failures on JetPack 6 |

**tinygrad with BEAM cache beats llama.cpp by ~32%.** A Python ML framework outperforming an optimized C++ inference engine — on an edge device.

## Understanding the Benchmark

### What Was Measured

- **Model**: Qwen3 1B with Q6_K quantization (~1.1 GB weights)
- **Task**: Autoregressive text generation (decode)
- **Metric**: tokens per second (tok/s) — higher is better
- **Hardware**: Jetson AGX Orin 64GB, SM 8.7, ~102 GB/s memory bandwidth

### Why tok/s?

For LLM inference, tok/s directly measures user-perceived speed. At 36 tok/s, you see ~4.5 words per second — feels responsive. At 7.6 tok/s, it feels sluggish.

### The Theoretical Maximum

LLM decode is **memory-bandwidth bound**. Each token reads the entire model from memory:

$$\text{max tok/s} = \frac{\text{bandwidth}}{\text{model size}} = \frac{102 \text{ GB/s}}{1.1 \text{ GB}} \approx 93 \text{ tok/s}$$

Our 36.71 tok/s achieves **39% of theoretical peak**. The gap comes from:
- Quantization/dequantization overhead (Q6_K)
- Non-matmul layers (layer norm, softmax, RoPE)
- Attention computation (not purely bandwidth-bound)
- Kernel launch overhead
- Memory access pattern inefficiencies

### llama.cpp Comparison

llama.cpp achieves 25.59–27.80 tok/s on the same hardware. It uses handwritten CUDA kernels with:
- Custom Q6_K dequantization kernels
- Fused operations
- Manual memory management

tinygrad generates all kernels at runtime through its compiler, yet beats llama.cpp by 32%. The key difference: BEAM search finds kernel configurations that hand-tuned code didn't explore.

## The Performance Curve

```
First run (cold):     BEAM searching...    → 7.59 tok/s
                      ↓ 2-5 minutes of BEAM compilation
Second run (cached):                       → 36.71 tok/s
                      ↓ all caches warm
Nth run (cached):                          → 36.71 tok/s (stable)
```

### Cache Layers

TinyGrad has multiple cache layers, each with different warmup costs:

| Cache | What | Cold Penalty | Warm Benefit |
|-------|------|-------------|-------------|
| **Schedule cache** | AST → ExecItem list | ~5 ms/schedule | ~0.1 ms |
| **Method cache** | AST → CompiledRunner | compilation time | instant |
| **BEAM cache** | AST → optimal Opt list | seconds per kernel | instant |
| **Disk cache** | Persists BEAM across runs | disk read | skip all BEAM |

The BEAM disk cache (`~/.cache/tinygrad/`) is the most important. Without it, every Python process restart triggers full BEAM search. With it, optimal kernels are loaded instantly.

```bash
# Force re-BEAM (delete cache)
rm -rf ~/.cache/tinygrad/

# The BEAM cache key includes the AST + device + beam width
# So BEAM=2 and BEAM=4 have separate caches
```

## Profiling TinyGrad

### DEBUG Flags

```bash
# Basic timing per kernel
DEBUG=1 NV=1 python3 my_model.py
# scheduled   42 kernels in    0.12 ms |  cache hit a3f7c1d2

# Per-kernel timing with shapes
DEBUG=2 NV=1 python3 my_model.py
# 0.012 ms  matmul(256,256,256)  mem:0.75 MB

# Full kernel optimization details
DEBUG=3 NV=1 python3 my_model.py
# Shows applied_opts for each kernel + colored shape

# Generated PTX source
DEBUG=4 NV=1 python3 my_model.py
```

### PROFILE Mode

```bash
PROFILE=1 NV=1 python3 my_model.py
# Generates a Chrome-compatible trace file
# Open in chrome://tracing or https://ui.perfetto.dev
```

This records GPU timestamps per kernel using hardware signals — microsecond-accurate.

### Quick Benchmark Script

```python
import time
from tinygrad import Tensor, Device

# Ensure NV backend
assert "NV" in Device.DEFAULT

# Warmup
x = Tensor.randn(2048, 2048)
y = Tensor.randn(2048)
for _ in range(5):
    (x @ y).realize()

# Benchmark
N = 100
start = time.perf_counter()
for _ in range(N):
    (x @ y).realize()
Device[Device.DEFAULT].synchronize()
elapsed = time.perf_counter() - start
print(f"{N/elapsed:.1f} matvecs/sec, {elapsed/N*1000:.3f} ms/op")
```

## The Roofline Model

The **roofline model** tells you whether a kernel is compute-bound or memory-bound:

```
                     Peak Compute (FLOPS)
                    ╱
Performance        ╱
(GFLOPS)  ────────╱───────────────── ← compute ceiling
                 ╱
                ╱
               ╱  ← memory bandwidth slope
              ╱
─────────────╱
            ╱
           Ridge point
```

- **Left of ridge**: Memory-bound (bandwidth bottleneck)
- **Right of ridge**: Compute-bound (FLOPS bottleneck)

For the Orin:
- Peak FP32: ~5.3 TFLOPS
- Peak FP16 (tensor cores): ~21 TFLOPS
- Bandwidth: ~102 GB/s
- Ridge point: 5300 / 102 ≈ **52 FLOPS/byte**

### Arithmetic Intensity

$$\text{AI} = \frac{\text{FLOPS}}{\text{bytes transferred}}$$

For matvec W[M,K] × x[K]:
- FLOPS: $2 \times M \times K$ (multiply + add)
- Bytes: $(M \times K + K) \times \text{bytes\_per\_element}$
- AI ≈ $\frac{2MK}{MK \times \text{bpe}} = \frac{2}{\text{bpe}}$

For Q6_K (6.5 bits/weight ≈ 0.81 bytes/weight):
- AI ≈ $\frac{2}{0.81} \approx 2.5$ FLOPS/byte

2.5 << 52 (ridge point) → **massively memory-bound**. This confirms that bandwidth optimization (matvec heuristic, vector loads, cache-friendly access) matters far more than compute optimization (tensor cores, instruction-level parallelism).

## What Matters for Performance

Ranked by impact on the Orin:

### 1. Matvec Heuristic (7.6×)

The [matvec fix](11-matvec-heuristic.md) was the single biggest win. Without it, GROUP_REDUCE never applied, and every matvec kernel ran with one thread per output row.

### 2. BEAM Search Cache (~4.8×)

BEAM empirically finds optimal UPCAST/LOCAL/GROUP combinations per kernel. The cache ensures this only happens once.

### 3. Kernel Fusion

TinyGrad fuses elementwise operations into preceding kernels:
```
Without fusion: matmul → write → read → relu → write → read → add
With fusion:    matmul+relu+add → write                        (1 kernel)
```
Fewer kernel launches = less overhead + better memory efficiency.

### 4. Vector Loads (UPCAST)

Loading 128 bits (4 floats) at once vs 32 bits (1 float):
```ptx
// 1 transaction
ld.global.v4.f32 {%f0, %f1, %f2, %f3}, [%rd0];

// vs 4 transactions
ld.global.f32 %f0, [%rd0];
ld.global.f32 %f1, [%rd0+4];
ld.global.f32 %f2, [%rd0+8];
ld.global.f32 %f3, [%rd0+12];
```

UPCAST enables this — processing 4 elements per thread means loading 4 at once.

### 5. Memory Planning

Reusing dead buffers ([Pill 6](06-schedule-memory.md)) reduces allocation overhead and improves cache locality by keeping working sets small.

### 6. Direct memmove on Tegra

Bypassing DMA staging for unified memory ([Pill 10](10-jetson-nv-backend-pt2.md)) eliminates unnecessary data copies during weight loading.

## What Doesn't Matter (Much)

- **Tensor cores**: Only help during prefill (matrix × matrix). Decode is matvec — tensor cores can't accelerate it.
- **Instruction-level optimizations**: We're memory-bound, not compute-bound. Faster math doesn't help when you're waiting for DRAM.
- **More SMs**: The Orin has 16 SMs. With good occupancy, we saturate the memory bus. More SMs would only help if we were compute-bound.

## Benchmarking Tips

### Consistent Results

```bash
# Pin GPU to max frequency (avoid governor downclocking)
echo $(cat /sys/class/devfreq/17000000.gpu/max_freq) > /sys/class/devfreq/17000000.gpu/min_freq

# Pin CPU to max
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable swap (OOM is better than swap thrashing for benchmarks)
swapoff -a

# Set NV power mode to max
nvpmodel -m 0
jetson_clocks
```

### Multiple Runs

```bash
# Run 5 times, report median
for i in $(seq 5); do
  NV=1 python3 bench.py 2>&1 | grep "tok/s"
done
```

### Compare BEAM Widths

```bash
# BEAM=0 (heuristic only)
rm -rf ~/.cache/tinygrad && BEAM=0 NV=1 python3 bench.py

# BEAM=2 (light search)
rm -rf ~/.cache/tinygrad && BEAM=2 NV=1 python3 bench.py

# BEAM=4 (medium search)
rm -rf ~/.cache/tinygrad && BEAM=4 NV=1 python3 bench.py
```

## Summary

- **36.71 tok/s** on Orin with tinygrad + BEAM cache — **beats llama.cpp by 32%**
- **Theoretical max**: 93 tok/s (memory bandwidth limit), we achieve 39%
- **BEAM cache** is critical — cold = 7.59, warm = 36.71 tok/s
- **LLM decode is memory-bound**: arithmetic intensity ~2.5 FLOPS/byte, ridge at 52
- **Performance ranking**: matvec heuristic > BEAM cache > fusion > vector loads
- **Tensor cores don't help** for decode (matvec, not matmul)
- **Pin frequencies** and use multiple runs for consistent benchmarks

---

**Previous**: [← Pill 11: The Matvec Heuristic](11-matvec-heuristic.md)
**Next**: [Pill 13: ML Primer →](13-ml-primer.md)
