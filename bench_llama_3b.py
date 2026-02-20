#!/usr/bin/env python3
"""
LLaMA 3B model benchmark suite for Jetson Orin AGX.
Tests LLaMA 3.2 at 1B and 3B scale with multiple quantization formats.
Includes BEAM search support (kernel cache speeds up subsequent runs).

Usage:
    NV=1 MV_THREADS_PER_ROW=32 python3 bench_llama_3b.py
    CUDA=1 MV_THREADS_PER_ROW=32 python3 bench_llama_3b.py
    NV=1 MV_THREADS_PER_ROW=32 JITBEAM=2 python3 bench_llama_3b.py
"""
import os, time, sys, gc, traceback

if __name__ != "__main__":
    raise RuntimeError("Run this script directly, not via import")

from tinygrad.apps.llm import Transformer, models
from tinygrad import Tensor, Device
from tinygrad.helpers import fetch

backend = "NV" if os.environ.get("NV") else "CUDA" if os.environ.get("CUDA") else "CPU"
mv_tpr = os.environ.get("MV_THREADS_PER_ROW", "8")
jitbeam = os.environ.get("JITBEAM", "0")

print(f"\n{'='*70}")
print(f"LLaMA 3B Benchmark - {backend}=1  MV_TPR={mv_tpr}  JITBEAM={jitbeam}")
print(f"Device: {Device.DEFAULT} | Jetson Orin AGX 64GB")
print(f"{'='*70}\n")

# All LLaMA models available in tinygrad with correct keys
llama_models = [
    ("llama3.2:1b",     "LLaMA 3.2 1B Q6_K",   "Q6_K"),
    ("llama3.2:1b-q4",  "LLaMA 3.2 1B Q4_K_M", "Q4_K_M"),
    ("llama3.2:3b",     "LLaMA 3.2 3B Q6_K",   "Q6_K"),
]

# Optional: full fp16 3B (6.1 GB, slow) — enable with BENCH_FP16=1
if os.environ.get("BENCH_FP16"):
    llama_models.append(("llama3.2:3b-f16", "LLaMA 3.2 3B fp16", "fp16"))

# Optional: 8B model — enable with BENCH_8B=1
if os.environ.get("BENCH_8B"):
    llama_models.append(("llama3.1:8b", "LLaMA 3.1 8B Q8_0", "Q8_0"))

results = []

for model_key, model_name, quant in llama_models:
    print(f"\n{'─'*70}")
    print(f"  {model_name}")
    print(f"{'─'*70}")

    if model_key not in models:
        print(f"  Model '{model_key}' not in tinygrad models dict, skipping")
        continue

    try:
        url = models[model_key]
        filename = url.rsplit("/", 1)[1]
        subdir = model_key.replace(":", "-")

        print(f"  Loading {filename}...", flush=True)
        gguf_path = fetch(url, filename, subdir=subdir)
        size_gb = os.path.getsize(gguf_path) / (1 << 30)
        print(f"  Size: {size_gb:.2f} GB")
    except Exception as e:
        print(f"  ERROR downloading: {e}")
        continue

    try:
        # Warmup (includes JIT compilation + BEAM cache build on first run)
        print(f"  Warmup...", flush=True)
        t_warm = time.time()
        model, kv = Transformer.from_gguf(Tensor(gguf_path))
        prompt = [128000, 9906]  # LLaMA-family BOS + token
        for i, tok in enumerate(model.generate(prompt)):
            if i >= 3: break
        warmup_s = time.time() - t_warm
        print(f"  Warmup: {warmup_s:.1f}s")
        del model, kv
        gc.collect()

        # Decode benchmark
        print(f"  Benchmarking decode...", flush=True)
        model, kv = Transformer.from_gguf(Tensor(gguf_path))
        times = []
        for i, tok in enumerate(model.generate(prompt)):
            times.append(time.time())
            if i >= 25: break

        if len(times) > 8:
            decode_times = [times[j] - times[j-1] for j in range(6, len(times))]
            avg_ms = sum(decode_times) / len(decode_times) * 1000
            tok_s = len(decode_times) / sum(decode_times)
            min_ms = min(decode_times) * 1000
            max_ms = max(decode_times) * 1000

            print(f"  Decode: {tok_s:6.2f} tok/s ({avg_ms:.1f}ms avg, {min_ms:.0f}-{max_ms:.0f}ms range)")
            results.append({
                "model": model_key, "name": model_name, "quant": quant,
                "size_gb": round(size_gb, 2), "tok_s": round(tok_s, 2),
                "avg_ms": round(avg_ms, 1), "warmup_s": round(warmup_s, 1),
            })
        else:
            print(f"  Not enough tokens generated ({len(times)})")

        del model, kv
        gc.collect()

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()

# ============= SUMMARY =============
print(f"\n{'='*70}")
print(f"SUMMARY  ({backend}=1  JITBEAM={jitbeam}  MV_TPR={mv_tpr})")
print(f"{'='*70}")

if results:
    print(f"\n{'Model':<28} {'Quant':<8} {'Size':>6} {'tok/s':>8} {'ms/tok':>8}")
    print("─" * 65)
    for r in results:
        print(f"{r['model']:<28} {r['quant']:<8} {r['size_gb']:5.1f}G {r['tok_s']:7.2f} {r['avg_ms']:7.1f}")

    # Quantization comparison: Q6_K vs Q4_K_M for same model size
    q6 = next((r for r in results if r["model"] == "llama3.2:1b"), None)
    q4 = next((r for r in results if r["model"] == "llama3.2:1b-q4"), None)
    if q6 and q4:
        delta = (q4["tok_s"] - q6["tok_s"]) / q6["tok_s"] * 100
        print(f"\nQuantization impact (LLaMA 1B):")
        print(f"  Q6_K: {q6['tok_s']:.1f} tok/s ({q6['size_gb']:.1f}G)")
        print(f"  Q4_K_M: {q4['tok_s']:.1f} tok/s ({q4['size_gb']:.1f}G) [{delta:+.1f}%]")

    # Scaling: 1B vs 3B
    m1b = next((r for r in results if r["model"] == "llama3.2:1b"), None)
    m3b = next((r for r in results if r["model"] == "llama3.2:3b"), None)
    if m1b and m3b:
        scale = m1b["tok_s"] / m3b["tok_s"]
        print(f"\nScaling (same quant Q6_K):")
        print(f"  1B: {m1b['tok_s']:.1f} tok/s ({m1b['size_gb']:.1f}G)")
        print(f"  3B: {m3b['tok_s']:.1f} tok/s ({m3b['size_gb']:.1f}G) [{scale:.1f}x slower]")
else:
    print("No results collected.")

print(f"\n{'='*70}\n")
