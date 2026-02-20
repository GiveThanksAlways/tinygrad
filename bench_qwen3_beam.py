#!/usr/bin/env python3
"""Benchmark Qwen3 models with JITBEAM on tinygrad NV/CUDA backends."""
import multiprocessing
multiprocessing.set_start_method("fork", force=True)

import os, time, sys

from tinygrad.apps.llm import Transformer, models
from tinygrad import Tensor, Device
from tinygrad.helpers import fetch

backend = "NV" if os.environ.get("NV") else "CUDA" if os.environ.get("CUDA") else "CPU"
beam = os.environ.get("JITBEAM", "0")

for model_name in ["qwen3:0.6b", "qwen3:1.7b"]:
    print(f"\n=== {model_name} {backend}=1 JITBEAM={beam} ===", flush=True)
    url = models[model_name]
    filename = url.rsplit("/", 1)[1]
    subdir = model_name.replace(":", "-")
    gguf = fetch(url, filename, subdir=subdir)
    
    model, kv = Transformer.from_gguf(Tensor(gguf))
    tokens = [151644, 8948, 2610]
    times = []
    t_start = time.time()
    for i, tok in enumerate(model.generate(list(tokens))):
        times.append(time.time())
        if i >= 30: break
    
    # Skip first 10 tokens (BEAM search warmup + JIT)
    decode_times = [times[j]-times[j-1] for j in range(11, len(times))]
    if decode_times:
        avg_ms = sum(decode_times)/len(decode_times) * 1000
        print(f"  Decode: {avg_ms:.1f}ms = {1000/avg_ms:.1f} tok/s", flush=True)
    else:
        print("  Not enough tokens generated", flush=True)
    del model, kv
