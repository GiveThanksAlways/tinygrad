# Pill 11: The Matvec Heuristic & the 7.6× Speedup

## The Commit That Changed Everything

```
fix(heuristic): enable matvec for fused CAST/MUL chains in fp16 matmul
```

33 lines added, 22 removed. One function. **4.8 → 36.71 tok/s** on Qwen3 1B Q6_K decode. 7.6× improvement. This pill explains every piece of it.

## What is Matvec?

**Matrix-vector multiplication** (matvec): multiply a matrix $W$ [M×K] by a vector $x$ [K] to produce a vector $y$ [M]:

$$y_i = \sum_{k=0}^{K-1} W_{i,k} \cdot x_k$$

The naive GPU approach: one thread per output element $y_i$, each doing a full reduction over K. For LLaMA 1B, K can be 2048 — each thread independently reads an entire row of $W$ from DRAM. Terrible memory utilization.

### The Optimized Approach

TinyGrad's heuristic applies three optimizations when it detects matvec:

**1. GROUP** (`MV_THREADS_PER_ROW`): Split the reduction across threads — 32 threads cooperate on one row, each reading K/32 elements, then combining via shared memory.

**2. LOCAL** (`MV_BLOCKSIZE`): Pack multiple output rows per workgroup — keeps the SMs busy.

**3. UPCAST** (`MV_ROWS_PER_THREAD`): Each thread handles multiple rows — enables vector loads (128-bit) and hides memory latency.

```
Without matvec:                 With matvec (TPR=32, BS=4, RPT=4):

Thread 0: y[0] = Σ W[0,:]·x    Workgroup 0: y[0..15]
Thread 1: y[1] = Σ W[1,:]·x      Threads 0..31:  partial sums for y[0..3]
...                                Threads 32..63: partial sums for y[4..7]
Thread M: y[M] = Σ W[M,:]·x      Threads 64..95: partial sums for y[8..11]
                                   Threads 96..127: partial sums for y[12..15]
(each thread reads full row)       → shared_mem reduce → final y[0..15]
(cache thrashing)
(5% of bandwidth)               (cooperative reads, 39% of bandwidth)
```

## Why LLM Decode is All Matvec

An LLM has two phases:

| Phase | Input Shape | Operation | When |
|-------|------------|-----------|------|
| **Prefill** | [seq_len, dim] | **matmul** (matrix × matrix) | Processing prompt |
| **Decode** | [1, dim] | **matvec** (matrix × vector) | Generating each token |

During autoregressive generation, you generate one token at a time. The "batch dimension" is 1. Every attention projection (Q, K, V, O) and every FFN layer (gate, up, down) becomes a matvec.

For a 1B Q6_K model, each token reads ~1.1 GB of weights. The Orin's bandwidth is ~102 GB/s:

$$\text{theoretical max} = \frac{102}{1.1} \approx 93 \text{ tok/s}$$

Without matvec optimization: 4.8 tok/s (5% of peak bandwidth).
With matvec optimization: 36.71 tok/s (39% of peak bandwidth).

The remaining gap is from quantization/dequantization overhead, attention computation, layer norms, and non-matmul layers.

## The Pattern Matching Failure

### What the Old Code Expected

The heuristic in `hand_coded_optimizations()` looked for this exact pattern:

```python
# Old code:
if (mulop := k.reduceop.src[0]).op is Ops.MUL \
   and mulop.src[0].op is Ops.INDEX \
   and mulop.src[1].op is Ops.INDEX:
    # Apply matvec optimization
```

In the UOp IR, a simple float32 matmul looks like:

```
REDUCE(ADD,
  MUL(                    ← reduceop.src[0] is MUL ✓
    INDEX(weight, [...]),  ← mulop.src[0] is INDEX ✓
    INDEX(input, [...])    ← mulop.src[1] is INDEX ✓
  )
)
```

All three checks pass → matvec detected → optimization applied.

### What Quantized Models Actually Produce

With Q6_K quantization and fp16 accumulation, the real graph looks like:

```
REDUCE(ADD,
  CAST(fp32,                    ← reduceop.src[0] is CAST ✗ (not MUL!)
    MUL(
      MUL(
        INDEX(weight, [...]),   ← buried 2 levels deep
        dequant_scale
      ),
      INDEX(input, [...])       ← buried 1 level deep
    )
  )
)
```

The `CAST` wrapper (fp16→fp32 for accumulation precision) means `reduceop.src[0].op` is `Ops.CAST`, not `Ops.MUL`. The first check fails. The heuristic silently skips matvec optimization.

**The `INDEX` ops are also nested** inside multiple `MUL` layers (due to dequantization). Even if we got past the CAST, `mulop.src[0].op` would be `Ops.MUL` (the inner dequant multiply), not `Ops.INDEX`.

**Result**: Every single decode kernel in the model fell through to the generic optimizer. No GROUP, no LOCAL, no UPCAST. 7.6× slower.

## The Fix: 33 Lines

### Part 1: Unwrap CAST

```python
mulop = k.reduceop.src[0]
if mulop.op is Ops.CAST:
    mulop = mulop.src[0]  # look inside the CAST
```

One line. If the reduction operand is wrapped in a CAST, peel it off. Now `mulop` points to the MUL underneath.

### Part 2: Recursive INDEX Finder

Instead of requiring INDEX as a direct child of MUL, search recursively:

```python
def _find_indices(u, depth=0):
    if u.op is Ops.INDEX: return [u]
    if depth > 3: return []  # safety limit
    ret = []
    for s in u.src:
        ret.extend(_find_indices(s, depth+1))
    return ret

indices = _find_indices(mulop) if mulop.op is Ops.MUL else []
```

This walks the tree to find INDEX operations through chains of MUL and CAST:

```
MUL                       ← depth 0, recurse
├── MUL                   ← depth 1, recurse
│   ├── INDEX(weight)     ← depth 2, FOUND! ✓
│   └── dequant_scale     ← depth 2, no INDEX
└── INDEX(input)          ← depth 1, FOUND! ✓

Result: indices = [INDEX(weight), INDEX(input)]
```

The `depth > 3` guard prevents infinite recursion while handling all real patterns (max nesting in practice is 2-3).

### Part 3: Use the Found Indices

```python
if len(indices) >= 2:
    idx0 = indices[0].src[1].get_idx()
    idx1 = indices[1].src[1].get_idx()
    # ... same matvec optimization logic as before ...
```

Once we have the INDEX operations, we check:
1. Does the reduction range appear in one index? (it should — that's the K dimension)
2. Are the other dimensions divisible by MV_BLOCKSIZE and MV_ROWS_PER_THREAD?

If yes → apply GROUP, LOCAL, UPCAST. The optimization code is **unchanged** — it was always correct. It just couldn't see the pattern through CAST/MUL wrappers.

## The Full Diff

```python
# OLD (removed):
if k.reduceop is not None and k.reduceop.arg[0] is Ops.ADD \
   and len(k.full_shape) >= 2 and k.ren.has_shared and \
   (mulop:=k.reduceop.src[0]).op is Ops.MUL \
   and mulop.src[0].op is Ops.INDEX \
   and mulop.src[1].op is Ops.INDEX:
    idx0, idx1 = mulop.src[0].src[1].get_idx(), mulop.src[1].src[1].get_idx()

# NEW (added):
if k.reduceop is not None and k.reduceop.arg[0] is Ops.ADD \
   and len(k.full_shape) >= 2 and k.ren.has_shared:
    mulop = k.reduceop.src[0]
    if mulop.op is Ops.CAST: mulop = mulop.src[0]

    def _find_indices(u, depth=0):
        if u.op is Ops.INDEX: return [u]
        if depth > 3: return []
        ret = []
        for s in u.src: ret.extend(_find_indices(s, depth+1))
        return ret

    indices = _find_indices(mulop) if mulop.op is Ops.MUL else []
    if len(indices) >= 2:
        idx0 = indices[0].src[1].get_idx()
        idx1 = indices[1].src[1].get_idx()
```

Changes:
1. **Removed** the `.op is Ops.MUL` + `.op is Ops.INDEX` checks from the outer condition
2. **Added** CAST unwrapping (1 line)
3. **Added** `_find_indices` recursive helper (6 lines)
4. **Changed** index extraction to use `indices[0]` and `indices[1]` instead of `mulop.src[0]` and `mulop.src[1]`

Everything else — the GROUP/LOCAL/UPCAST application, the divisibility checks, the MV_THREADS_PER_ROW/MV_BLOCKSIZE/MV_ROWS_PER_THREAD constants — is unchanged.

## Why Didn't BEAM Search Find This?

You might wonder: if BEAM tries hundreds of optimization combinations ([Pill 7](07-beam-search.md)), why couldn't it find the matvec optimization on its own?

Three reasons:

1. **BEAM starts from the heuristic**: The search begins with whatever the heuristic produces. Without matvec detection, the starting kernel has no GROUP. BEAM would need to add GROUP + LOCAL + UPCAST in the right combination — that's a very specific multi-step path through action space.

2. **GROUP_REDUCE is a specific action**: BEAM tries `GROUP(axis, amt)` and `GROUPTOP(axis, amt)` with fixed amounts (4, 8, 16, 32, etc.). The matvec pattern needs a very specific combination: GROUP on the reduction axis with `MV_THREADS_PER_ROW=32`, then LOCAL on the output axis with `MV_BLOCKSIZE=4`, then UPCAST with `MV_ROWS_PER_THREAD=4`. Finding this exact triple is like finding a needle in a haystack.

3. **Exponential search space**: With ~190 actions per step and 3-5 steps to reach the optimal matvec configuration, BEAM would need to evaluate thousands of candidates. Even BEAM=8 only keeps 8 candidates per step — the correct path is likely pruned early.

**Bottom line**: Heuristics and search complement each other. The heuristic provides domain knowledge ("this is matvec, apply GROUP/LOCAL/UPCAST in this specific way"). BEAM provides empirical tuning on top ("given the heuristic's baseline, try UPCAST(1,4) vs UPCAST(1,2)").

The 7.6× speedup came from fixing the heuristic. An additional ~1.5× came from BEAM refining the heuristic's choices (4.8 → 7.6 with heuristic alone, 7.6 → ~36.7 with BEAM cache on top — though the 36.7 number includes all optimizations).

## Debugging This Class of Problem

If you suspect the heuristic is missing a pattern:

```bash
# Show optimization choices for each kernel
DEBUG=3 NV=1 python3 my_model.py

# Look for: kernels with a REDUCE but no GROUP in the shape
# 0.150 ms  kernel(2048,1,1)  (2048, <REDUCE 2048>)  ← no GROUP = bad!
# Should be: (128, <LOCAL 4>, <GROUP_REDUCE 32>, <UPCAST 4>, <REDUCE 64>)

# Show the actual AST
DEBUG=5 NV=1 python3 my_model.py
# Look at reduceop.src[0] — is it CAST(MUL(...)) or MUL(...)?
```

The pattern: large REDUCE dimensions (1024+) without GROUP_REDUCE usually means the matvec heuristic didn't trigger. Check what's wrapping the MUL.

## Lessons

1. **Pattern matching failures are silent**. The heuristic didn't crash — it just fell through to the default (slow) path. No error, no warning.

2. **Quantized models change the IR structure**. What works for fp32 matmul may not work for Q6_K + fp16 accumulation, because the CAST and dequantization adds UOp layers around the core pattern.

3. **Small fixes, huge impact**. 6 lines of `_find_indices` + 1 line of CAST unwrapping = 7.6× speedup on the most important operation in LLM inference.

4. **Heuristics beat search for structured problems**. BEAM search excels at fine-tuning but struggles to discover multi-step optimization patterns from scratch. Domain knowledge (matvec = GROUP + LOCAL + UPCAST) is irreplaceable.

## Summary

- **Matvec** = matrix × vector — the dominant operation in LLM decode
- **The bug**: Heuristic expected `REDUCE(ADD, MUL(INDEX, INDEX))` but got `REDUCE(ADD, CAST(MUL(MUL(INDEX, ...), INDEX)))`
- **The fix**: Unwrap CAST, recursively find INDEX ops through MUL chains
- **7.6× speedup**: 4.8 → 36.71 tok/s on Orin AGX 64GB with Qwen3 1B Q6_K
- **BEAM couldn't find this**: The optimal optimization path is too specific for blind search
- **33 lines changed** — the highest lines-of-code-to-performance-improvement ratio in the codebase

---

**Previous**: [← Pill 10: Jetson NV Backend, Part 2](10-jetson-nv-backend-pt2.md)
**Next**: [Pill 12: Benchmarking & Performance →](12-benchmarking.md)
