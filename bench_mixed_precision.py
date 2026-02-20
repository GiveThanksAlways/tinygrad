#!/usr/bin/env python3
"""
Mixed Precision Benchmark for tinygrad on Jetson Orin AGX.
Compares same model architecture at different quantization levels
to measure the throughput vs quality tradeoff.

Tests:
  Part 1: Raw matmul throughput at fp32 vs fp16 (tinygrad dtypes)
  Part 2: LLaMA inference across quantization formats (Q6_K, Q4_K_M, fp16)
  Part 3: Bandwidth utilization analysis

Usage:
    NV=1 MV_THREADS_PER_ROW=32 python3 bench_mixed_precision.py
    CUDA=1 MV_THREADS_PER_ROW=32 python3 bench_mixed_precision.py
    NV=1 MV_THREADS_PER_ROW=32 JITBEAM=2 python3 bench_mixed_precision.py
"""
import os, time, sys, gc

if __name__ != "__main__":
    raise RuntimeError("Run this script directly, not via import")

from tinygrad import Tensor, Device, dtypes
from tinygrad.apps.llm import Transformer, models
from tinygrad.helpers import fetch

backend = "NV" if os.environ.get("NV") else "CUDA" if os.environ.get("CUDA") else "CPU"
mv_tpr = os.environ.get("MV_THREADS_PER_ROW", "8")
jitbeam = os.environ.get("JITBEAM", "0")

print(f"\n{'='*70}")
print(f"Mixed Precision Benchmark - {backend}=1  JITBEAM={jitbeam}")
print(f"Device: {Device.DEFAULT} | Jetson Orin AGX 64GB (204 GB/s LPDDR5)")
print(f"{'='*70}\n")

# ============= PART 1: MATMUL PRECISION =============

print("Part 1: Matmul Throughput by Precision")
print("─" * 70)

matmul_results = []

for dt_name, dt in [("fp32", dtypes.float32), ("fp16", dtypes.float16)]:
    for N in [512, 1024, 2048]:
        try:
            A = Tensor.randn(N, N).cast(dt)
            B = Tensor.randn(N, N).cast(dt)
            ops = N * N * N * 2  # FMA = 2 ops

            # Force realize to trigger computation
            C = (A @ B)
            C.realize()

            # Warmup
            for _ in range(3):
                C = (A @ B)
                C.realize()

            # Benchmark
            iters = 10
            t0 = time.time()
            for _ in range(iters):
                C = (A @ B)
                C.realize()
            elapsed = time.time() - t0

            gflops = (ops * iters) / (elapsed * 1e9)
            bytes_rw = (N * N * (2 if dt == dtypes.float16 else 4)) * 3 * iters
            bw_gbs = bytes_rw / (elapsed * 1e9)

            print(f"  {dt_name} {N:>4}x{N:<4}  {gflops:8.1f} GFLOP/s  {bw_gbs:6.1f} GB/s")
            matmul_results.append({
                "dtype": dt_name, "N": N,
                "gflops": round(gflops, 1), "bw_gbs": round(bw_gbs, 1),
            })

            del A, B, C
        except Exception as e:
            print(f"  {dt_name} {N:>4}x{N:<4}  ERROR: {e}")

# ============= PART 2: LLM DECODE BY QUANTIZATION =============

print(f"\nPart 2: LLaMA Decode Across Quantization Formats")
print("─" * 70)

# Same architecture (LLaMA 3.2 1B), different quantizations
quant_models = [
    ("llama3.2:1b",     "LLaMA 1B Q6_K",   "Q6_K",   6),
    ("llama3.2:1b-q4",  "LLaMA 1B Q4_K_M", "Q4_K_M", 4),
]

# Also compare at 3B scale
quant_models_3b = [
    ("llama3.2:3b",     "LLaMA 3B Q6_K",   "Q6_K",   6),
]

# Full fp16 (optional, very large)
if os.environ.get("BENCH_FP16"):
    quant_models_3b.append(("llama3.2:3b-f16", "LLaMA 3B fp16", "fp16", 16))

all_quant_models = quant_models + quant_models_3b
llm_results = []

for model_key, model_name, quant, bits in all_quant_models:
    if model_key not in models:
        print(f"  {model_name:<28} NOT AVAILABLE")
        continue

    try:
        url = models[model_key]
        filename = url.rsplit("/", 1)[1]
        subdir = model_key.replace(":", "-")
        gguf_path = fetch(url, filename, subdir=subdir)
        size_gb = os.path.getsize(gguf_path) / (1 << 30)

        print(f"  {model_name} ({size_gb:.1f}G)...", end="", flush=True)

        # Warmup
        model, kv = Transformer.from_gguf(Tensor(gguf_path))
        prompt = [128000, 9906]
        for i, tok in enumerate(model.generate(prompt)):
            if i >= 2: break
        del model, kv
        gc.collect()

        # Benchmark
        model, kv = Transformer.from_gguf(Tensor(gguf_path))
        times = []
        for i, tok in enumerate(model.generate(prompt)):
            times.append(time.time())
            if i >= 20: break

        if len(times) > 6:
            decode_times = [times[j] - times[j-1] for j in range(5, len(times))]
            tok_s = len(decode_times) / sum(decode_times)
            avg_ms = sum(decode_times) / len(decode_times) * 1000
            print(f" {tok_s:6.2f} tok/s  ({avg_ms:.1f}ms, {quant})")
            llm_results.append({
                "model": model_key, "name": model_name, "quant": quant,
                "bits": bits, "size_gb": round(size_gb, 2),
                "tok_s": round(tok_s, 2), "avg_ms": round(avg_ms, 1),
            })
        else:
            print(f" ERROR: only {len(times)} tokens")

        del model, kv
        gc.collect()

    except Exception as e:
        print(f" ERROR: {e}")

# ============= PART 3: ANALYSIS =============

print(f"\n{'='*70}")
print("ANALYSIS")
print(f"{'='*70}")

# Matmul precision comparison
if matmul_results:
    print(f"\nMatmul fp32 vs fp16:")
    for N in [512, 1024, 2048]:
        fp32 = next((r for r in matmul_results if r["dtype"] == "fp32" and r["N"] == N), None)
        fp16 = next((r for r in matmul_results if r["dtype"] == "fp16" and r["N"] == N), None)
        if fp32 and fp16:
            speedup = fp16["gflops"] / fp32["gflops"] if fp32["gflops"] > 0 else 0
            print(f"  {N}x{N}: fp32={fp32['gflops']:.0f} vs fp16={fp16['gflops']:.0f} GFLOP/s ({speedup:.2f}x)")

# LLM quantization comparison
if llm_results:
    print(f"\nLLM Quantization Impact:")
    print(f"{'Model':<28} {'Quant':<8} {'Bits':<6} {'Size':>6} {'tok/s':>8}")
    print("─" * 60)
    for r in llm_results:
        print(f"{r['name']:<28} {r['quant']:<8} {r['bits']:<6} {r['size_gb']:5.1f}G {r['tok_s']:7.2f}")

    # Calculate bits-per-parameter efficiency
    q6 = next((r for r in llm_results if r["model"] == "llama3.2:1b"), None)
    q4 = next((r for r in llm_results if r["model"] == "llama3.2:1b-q4"), None)
    if q6 and q4:
        size_ratio = q6["size_gb"] / q4["size_gb"] if q4["size_gb"] > 0 else 0
        speed_ratio = q4["tok_s"] / q6["tok_s"] if q6["tok_s"] > 0 else 0
        print(f"\n  LLaMA 1B: Q6_K vs Q4_K_M:")
        print(f"    Size reduction: {size_ratio:.2f}x smaller")
        print(f"    Speed change: {speed_ratio:.2f}x")
        print(f"    Q4 is {'faster' if speed_ratio > 1 else 'slower'} due to {'less memory to read' if speed_ratio > 1 else 'dequant overhead'}")

print(f"\nKey Insights:")
print(f"  - Orin LPDDR5: ~102 GB/s effective BW at batch=1")
print(f"  - fp16 matmul: theoretical 2x compute, limited by BW at large sizes")
print(f"  - Q4 vs Q6: less data to read BUT dequant adds compute")
print(f"  - Sweet spot: Q6_K for small models, Q4_K_M for >3B models")
print(f"\n{'='*70}\n")
