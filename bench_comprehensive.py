#!/usr/bin/env python3
"""
Comprehensive tinygrad benchmark: LLaMA 3B, mixed precision, BEAM search.
Tests multiple model sizes, quantization formats, and BEAM configurations.

This script is designed to be run as a FILE (not stdin) so that
multiprocessing spawn works correctly for BEAM search.

The first run with JITBEAM builds the kernel cache (slow).
Subsequent runs use cached kernels (fast).

Usage:
    # Basic (no BEAM)
    NV=1 MV_THREADS_PER_ROW=32 python3 bench_comprehensive.py

    # With BEAM search (first run: slow cache build; rerun: fast)
    NV=1 MV_THREADS_PER_ROW=32 JITBEAM=2 python3 bench_comprehensive.py
    NV=1 MV_THREADS_PER_ROW=32 JITBEAM=4 python3 bench_comprehensive.py

    # Control parallelism (experiment to find best)
    NV=1 MV_THREADS_PER_ROW=32 JITBEAM=2 PARALLEL=4 python3 bench_comprehensive.py
    NV=1 MV_THREADS_PER_ROW=32 JITBEAM=2 PARALLEL=0 python3 bench_comprehensive.py

    # CUDA backend
    CUDA=1 MV_THREADS_PER_ROW=32 JITBEAM=2 python3 bench_comprehensive.py

Environment Variables:
    NV=1             Use Tegra NV backend (recommended)
    CUDA=1           Use CUDA driver backend
    MV_THREADS_PER_ROW=32   Matvec thread config (use 32)
    JITBEAM=N        BEAM search depth (0=off, 2=fast, 4=thorough)
    PARALLEL=N       Workers for BEAM (0=single-thread, auto=cpu_count)
    HALF=1           Force fp16 (model-dependent)
"""
import os, time, sys, gc, json
from datetime import datetime

if __name__ != "__main__":
    raise RuntimeError("This script must be run directly (not imported)")

from tinygrad.apps.llm import Transformer, models
from tinygrad import Tensor, Device
from tinygrad.helpers import fetch

# ============= CONFIGURATION =============

backend = "NV" if os.environ.get("NV") else "CUDA" if os.environ.get("CUDA") else "CPU"
mv_tpr = os.environ.get("MV_THREADS_PER_ROW", "8")
jitbeam = os.environ.get("JITBEAM", "0")
parallel = os.environ.get("PARALLEL", "auto")

print(f"\n{'='*70}")
print(f"Comprehensive tinygrad Benchmark Suite")
print(f"{'='*70}")
print(f"Backend: {backend}=1")
print(f"MV_THREADS_PER_ROW: {mv_tpr}")
print(f"JITBEAM: {jitbeam}")
print(f"PARALLEL: {parallel}")
print(f"Device: {Device.DEFAULT}")
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*70}\n")

# ============= MODEL DEFINITIONS =============

# Models to benchmark: (key, display_name, prompt_tokens, quant_info)
benchmark_models = [
    # Small models (fast, good for validating BEAM cache)
    ("llama3.2:1b", "LLaMA 3.2 1B Q6_K", [128000, 9906], "Q6_K"),

    # 3B models (main target)
    ("llama3.2:3b", "LLaMA 3.2 3B Q6_K", [128000, 9906], "Q6_K"),

    # Mixed precision: same model family, different quantizations
    ("llama3.2:1b-q4", "LLaMA 3.2 1B Q4_K_M", [128000, 9906], "Q4_K_M"),

    # Qwen family
    ("qwen3:0.6b", "Qwen3 0.6B Q8_0", [151644, 8948], "Q8_0"),
    ("qwen3:1.7b", "Qwen3 1.7B Q4_K_M", [151644, 8948], "Q4_K_M"),
]

# 3B fp16 — only run if specifically requested (it's large and slow)
if os.environ.get("BENCH_FP16"):
    benchmark_models.append(
        ("llama3.2:3b-f16", "LLaMA 3.2 3B fp16", [128000, 9906], "fp16")
    )

# 8B model — only run if specifically requested (very large)
if os.environ.get("BENCH_8B"):
    benchmark_models.append(
        ("llama3.1:8b", "LLaMA 3.1 8B Q8_0", [128000, 9906], "Q8_0")
    )

# ============= BENCHMARK FUNCTION =============

def benchmark_model(model_key, model_name, prompt_tokens, quant_info,
                    warmup_tokens=3, decode_tokens=20, skip_tokens=5):
    """Benchmark a single model. Returns dict with results or None on failure."""

    print(f"\n{'─'*70}")
    print(f"  {model_name} ({backend})")
    print(f"{'─'*70}")

    if model_key not in models:
        print(f"  ⚠ Model '{model_key}' not available in tinygrad, skipping")
        return None

    try:
        # Download
        url = models[model_key]
        filename = url.rsplit("/", 1)[1]
        subdir = model_key.replace(":", "-")
        print(f"  Loading: {filename}", flush=True)
        t_dl = time.time()
        gguf_path = fetch(url, filename, subdir=subdir)
        dl_time = time.time() - t_dl
        model_size_gb = os.path.getsize(gguf_path) / (1 << 30)
        print(f"  Size: {model_size_gb:.2f} GB (fetch: {dl_time:.1f}s)")
    except Exception as e:
        print(f"  ERROR downloading: {e}")
        return None

    try:
        # === Warmup phase ===
        print(f"  Warmup ({warmup_tokens} tokens)...", flush=True)
        t_warmup_start = time.time()
        model, kv = Transformer.from_gguf(Tensor(gguf_path))
        for i, tok in enumerate(model.generate(list(prompt_tokens))):
            if i >= warmup_tokens:
                break
        warmup_time = time.time() - t_warmup_start
        print(f"  Warmup done: {warmup_time:.1f}s (includes JIT + BEAM cache build)")
        del model, kv
        gc.collect()

        # === Decode benchmark ===
        print(f"  Benchmarking decode ({decode_tokens} tokens)...", flush=True)
        model, kv = Transformer.from_gguf(Tensor(gguf_path))

        times = []
        for i, tok in enumerate(model.generate(list(prompt_tokens))):
            times.append(time.time())
            if i >= decode_tokens:
                break

        del model, kv
        gc.collect()

        if len(times) <= skip_tokens + 2:
            print(f"  ⚠ Only {len(times)} tokens generated, not enough for measurement")
            return None

        # Calculate decode metrics (skip first N tokens for stable measurement)
        decode_times = [times[j] - times[j-1] for j in range(skip_tokens + 1, len(times))]
        if not decode_times:
            print(f"  ⚠ No decode times after skipping {skip_tokens} tokens")
            return None

        avg_ms = sum(decode_times) / len(decode_times) * 1000
        tok_s = len(decode_times) / sum(decode_times)
        min_ms = min(decode_times) * 1000
        max_ms = max(decode_times) * 1000

        # First token time (includes prefill)
        ttft = (times[0] - (times[0] - decode_times[0])) if decode_times else 0

        print(f"  ✓ Decode: {tok_s:6.2f} tok/s ({avg_ms:.1f}ms avg, {min_ms:.1f}-{max_ms:.1f}ms range)")

        result = {
            "model": model_key,
            "name": model_name,
            "backend": backend,
            "quant": quant_info,
            "jitbeam": jitbeam,
            "parallel": parallel,
            "mv_tpr": mv_tpr,
            "tok_s": round(tok_s, 2),
            "avg_ms": round(avg_ms, 1),
            "min_ms": round(min_ms, 1),
            "max_ms": round(max_ms, 1),
            "model_size_gb": round(model_size_gb, 2),
            "warmup_s": round(warmup_time, 1),
            "tokens_measured": len(decode_times),
        }
        return result

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============= RUN BENCHMARKS =============

all_results = []
total_start = time.time()

for model_key, model_name, prompt_tokens, quant_info in benchmark_models:
    result = benchmark_model(model_key, model_name, prompt_tokens, quant_info)
    if result:
        all_results.append(result)
    sys.stdout.flush()

total_time = time.time() - total_start

# ============= RESULTS TABLE =============

print(f"\n{'='*70}")
print(f"RESULTS SUMMARY")
print(f"{'='*70}")
print(f"Backend: {backend}=1, JITBEAM={jitbeam}, MV_TPR={mv_tpr}")
print(f"Total time: {total_time:.0f}s\n")

if all_results:
    # Header
    print(f"{'Model':<28} {'Quant':<8} {'Size':>6} {'tok/s':>8} {'ms/tok':>8} {'Range':>14} {'Warmup':>8}")
    print("─" * 90)

    for r in all_results:
        range_str = f"{r['min_ms']:.0f}-{r['max_ms']:.0f}ms"
        print(f"{r['model']:<28} {r['quant']:<8} {r['model_size_gb']:5.1f}G {r['tok_s']:7.2f} {r['avg_ms']:7.1f} {range_str:>14} {r['warmup_s']:7.1f}s")

    # Mixed precision comparison
    q6k_results = [r for r in all_results if "Q6_K" in r["quant"]]
    q4k_results = [r for r in all_results if "Q4_K" in r["quant"]]

    if q6k_results and q4k_results:
        print(f"\n{'─'*70}")
        print("Mixed Precision Analysis:")
        for q6 in q6k_results:
            for q4 in q4k_results:
                if "llama3.2:1b" in q6["model"] and "llama3.2:1b" in q4["model"]:
                    delta = (q4["tok_s"] - q6["tok_s"]) / q6["tok_s"] * 100
                    print(f"  LLaMA 1B: Q6_K={q6['tok_s']:.1f} vs Q4_K_M={q4['tok_s']:.1f} tok/s ({delta:+.1f}%)")
                    print(f"  Memory: Q6_K={q6['model_size_gb']:.1f}G vs Q4_K_M={q4['model_size_gb']:.1f}G")

    # Model scaling analysis
    llama_results = [r for r in all_results if "llama" in r["model"].lower()]
    if len(llama_results) >= 2:
        print(f"\n{'─'*70}")
        print("Model Scaling:")
        for r in sorted(llama_results, key=lambda x: x["model_size_gb"]):
            print(f"  {r['model']:<25} {r['model_size_gb']:5.1f}G → {r['tok_s']:6.2f} tok/s")

    # BEAM impact note
    if jitbeam != "0":
        print(f"\n{'─'*70}")
        print(f"BEAM Search: JITBEAM={jitbeam}, PARALLEL={parallel}")
        print(f"  Note: First run builds kernel cache (warmup times include BEAM search).")
        print(f"  Subsequent runs use cached kernels and warmup will be much faster.")
        print(f"  To see BEAM speedup: re-run this script (cache already populated).")

    # JSON output for programmatic use
    print(f"\n{'─'*70}")
    print(f"JSON:")
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "backend": backend,
            "jitbeam": jitbeam,
            "parallel": parallel,
            "mv_tpr": mv_tpr,
        },
        "results": all_results,
        "total_time_s": round(total_time, 1),
    }
    print(json.dumps(output, indent=2))

else:
    print("No results collected. Check model availability and GPU access.")

print(f"\n{'='*70}")
print(f"Benchmark complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*70}\n")
