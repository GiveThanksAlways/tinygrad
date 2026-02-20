# Pill 7: BEAM Search — Finding Fast Kernels

## The Optimization Problem

In [Pill 2](02-compilation-pipeline.md), you learned that TinyGrad compiles high-level tensor ops into GPU kernels. But the same computation can be organized many ways:

```
Matrix multiply: C[256,256] = A[256,256] × B[256,256]

Option A: 1 thread per output element, simple loop
  → global_size=(256,256,1), local_size=(1,1,1)
  → 1.2 ms

Option B: 4x upcast, 16-thread local group
  → global_size=(64,256,1), local_size=(16,1,1)
  → 0.3 ms

Option C: Tensor cores, 16x16 tiles, shared memory
  → global_size=(16,16,1), local_size=(128,1,1)
  → 0.05 ms
```

24× difference between worst and best. **BEAM search finds option C automatically.**

## What is BEAM Search?

BEAM search is a **bounded breadth-first search** over kernel optimization choices. Instead of trying every possible combination (exponential), it keeps only the `N` best candidates at each step.

```
             Start: unoptimized kernel
                     │
          ┌──────────┼──────────┐
          │          │          │
       UPCAST(0,4) LOCAL(0,16) TC(0)
          │          │          │
       0.8ms      0.5ms     0.3ms   ← time each on real hardware
          │          │          │
     ┌────┘    ┌────┘    ┌────┘
     │  keep   │  keep   │  keep    ← keep top N=3
     │  beam   │  beam   │  beam
     ▼         ▼         ▼
   try more  try more  try more
   actions   actions   actions
     │         │         │
     ...       ...       ...

   Repeat until no improvement → return best
```

The name "BEAM" comes from the beam width — the number of candidates kept at each step. `BEAM=2` keeps 2 candidates, `BEAM=8` keeps 8.

## The Action Space

BEAM search explores a set of **optimization actions**. Each action transforms the kernel's structure:

```python
actions = [
    # UPCAST: process multiple elements per thread (vectorization)
    Opt(OptOps.UPCAST, axis=0, arg=4),   # 4 elements per thread on axis 0
    Opt(OptOps.UPCAST, axis=1, arg=2),   # 2 elements per thread on axis 1

    # UNROLL: unroll loop iterations
    Opt(OptOps.UNROLL, axis=0, arg=4),   # unroll 4 iterations on axis 0

    # LOCAL: assign threads to a local work group (block)
    Opt(OptOps.LOCAL, axis=0, arg=16),   # 16 threads locally on axis 0

    # GROUPTOP / GROUP: restructure reduction operations
    Opt(OptOps.GROUPTOP, axis=0, arg=32),  # group reduction into 32-element chunks

    # TC: enable tensor cores
    Opt(OptOps.TC, axis=0, arg=(-1, 2, 1)),  # tensor core on axis 0

    # SWAP: reorder axes (loop ordering)
    Opt(OptOps.SWAP, axis=0, arg=1),     # swap axis 0 and 1

    # THREAD: like LOCAL but for non-local threading
    Opt(OptOps.THREAD, axis=0, arg=32),  # 32 threads

    # PADTO: pad a dimension to a multiple
    Opt(OptOps.PADTO, axis=0, arg=32),   # pad axis 0 to multiple of 32
]
```

There are hundreds of possible actions. For reference, the actual action list has:
- 48 UPCAST variants (6 amounts × 8 axes)
- 15 UNROLL variants (3 amounts × 5 axes)
- 44 LOCAL variants (7 amounts × 6 axes + 2 extras)
- 24 GROUPTOP variants
- 12 GROUP variants
- 9 TC variants
- 10 SWAP variants
- 30 THREAD variants

Total: **~190+ possible actions** per step.

## How BEAM Search Works

### Step 1: Generate Candidates

For each kernel in the current beam, try every valid action:

```python
def get_kernel_actions(s: Scheduler, include_0=True) -> dict[int, Scheduler]:
    acted = {0: s} if include_0 else {}
    for i, action in enumerate(actions):
        s2 = s.copy()
        try:
            s2.apply_opt(action)
            # Filter: too many upcasts or too many local threads?
            up, lcl = compute_upcast_and_local(s2)
            if up > BEAM_UPCAST_MAX or lcl > BEAM_LOCAL_MAX:
                continue
            acted[i+1] = s2
        except KernelOptError:
            pass  # invalid action for this kernel → skip
    return acted
```

Not all actions are valid. `apply_opt` raises `KernelOptError` if:
- The axis doesn't exist (e.g., UPCAST axis 7 on a 3D kernel)
- The amount doesn't divide the dimension evenly
- The action would violate hardware limits (too many registers, too much shared memory)

### Step 2: Compile Candidates (Parallel)

Each candidate is compiled in a worker process:

```python
def _try_compile(x: tuple[int, Scheduler], compiler: Compiler):
    p = get_program(x[1].get_optimized_ast(), x[1].ren)  # lower → render → PTX

    # Sanity check: too many UOps?
    if len(p.uops) >= BEAM_UOPS_MAX:
        raise RuntimeError("too many uops")

    # Compile PTX → CUBIN
    prog = compiler.compile(p.src)
    return (p, prog, compile_time)
```

**Multiprocessing**: BEAM uses a process pool for compilation:

```python
if beam_pool is None and workers > 0:
    _mp_ctx = "fork" if sys.platform == "linux" else "spawn"
    beam_pool = multiprocessing.get_context(_mp_ctx).Pool(workers, _init_worker)
```

On Linux, it uses `fork` (faster — shares memory). On other platforms, `spawn` (safer — clean processes). Our fork-vs-spawn fix (covered in [Pill 10](10-jetson-nv-backend-pt2.md)) ensures `fork` works correctly on Tegra.

Workers are initialized with device access disabled (`ALLOW_DEVICE_USAGE=0`) and SIGINT ignored — they're pure compilation workers, not GPU users.

### Step 3: Time on Real Hardware

Each compiled kernel is benchmarked on the actual GPU:

```python
def _time_program(p, lib, var_vals, rawbufs, early_stop, cnt=3):
    car = CompiledRunner(p_with_lib)

    tms = []
    for _ in range(cnt):                     # run 3 times
        if clear_l2:
            dev.invalidate_caches()          # cold cache for fair comparison
        tm = car(input_bufs, var_vals, wait=True)  # launch + sync + measure
        tms.append(tm * factor)
        if early_stop and early_stop < min(tms):
            break                            # already worse than best → skip
    return tms
```

Key details:
- **3 runs, take the minimum**: reduces noise from scheduling jitter
- **Early stop**: if this candidate is already 3× slower than the best, stop timing
- **Factor scaling**: if the grid was shrunk for testing, multiply time back up
- **L2 cache clearing**: optional, for fair memory-bound comparisons
- **Test global size**: large grids are shrunk to fit `max_global_size=65536` for faster testing

### Step 4: Select Best, Repeat or Exit

```python
while not exiting:
    candidates = flatten([get_kernel_actions(si).values() for si, _ in beam])
    # ... compile and time all candidates ...

    opts = sorted(timed, key=lambda x: x[1])  # sort by time

    exiting = (
        len(opts) == 0 or                          # no valid candidates
        opts[0][1] < min_progress or                # already near zero (0.01 µs)
        (beam[0][1] - opts[0][1]) < min_progress    # improvement too small
    )

    if not exiting:
        beam = opts[:amt]  # keep top N candidates
    elif opts[0][1] < beam[0][1]:
        beam = opts[:1]    # one final improvement
```

Exit conditions:
1. **No valid actions left** — kernel is fully optimized
2. **Already very fast** — below 10 ns, can't improve
3. **Diminishing returns** — improvement less than 10 ns per step

## The BEAM Cache

BEAM search is expensive — timing 100+ kernel variants on real hardware takes seconds. The results are cached to disk:

```python
# Check cache before searching
key = {"ast": s.ast.key, "amt": amt, "device": s.ren.device, ...}
if (val := diskcache_get("beam_search", key)) is not None:
    # Apply cached optimization sequence
    ret = s.copy()
    for o in val[len(s.applied_opts):]:
        ret.apply_opt(o)
    return ret

# After search: save to cache
diskcache_put("beam_search", key, beam[0][0].applied_opts)
```

The cache key includes the kernel AST + device + beam width. The value is the list of `Opt` objects to apply.

**This is why the first run is slow but subsequent runs are fast.** On the Orin:
- First run (cold cache): 7.59 tok/s (BEAM searching every kernel)
- Subsequent runs (warm cache): 36.71 tok/s (BEAM results loaded instantly)

The cache lives in `~/.cache/tinygrad/` — delete it to force re-optimization.

## Using BEAM

### Environment Variables

```bash
# Enable BEAM with width 2 (keep 2 candidates per step)
BEAM=2 NV=1 python3 my_model.py

# BEAM with width 8 (slower search, potentially better results)
BEAM=8 NV=1 python3 my_model.py

# JITBEAM: apply BEAM to JIT-compiled functions (re-optimizes on shape changes)
JITBEAM=2 NV=1 python3 my_model.py

# Debug BEAM decisions
BEAM_DEBUG=1 BEAM=2 NV=1 python3 my_model.py
```

### BEAM vs Heuristic

TinyGrad has two optimization strategies:

| | Heuristic | BEAM |
|-----|-----------|------|
| Speed | Instant (~0 ms) | Slow (seconds per kernel) |
| Quality | Good (hand-tuned rules) | Great (empirically optimal) |
| When | Default (BEAM=0) | Opt-in (BEAM=N) |
| Caching | Not needed | Disk cache critical |
| Code | `heuristic.py` | `search.py` |

The **heuristic** (`hand_coded_optimizations()` in heuristic.py) applies rule-based optimizations:
- Detect tensor core eligibility → apply TC
- Detect matvec pattern → set GROUP_REDUCE + UPCAST
- Apply LOCAL based on shape analysis
- These run in microseconds

**BEAM** overrides the heuristic by empirically timing every option. It finds combinations the heuristic misses — like unusual UPCAST/LOCAL/UNROLL combos that happen to hit the memory system well.

For LLM inference, BEAM typically improves throughput 2-5× over heuristic alone.

### BEAM Output Example

```bash
$ BEAM=2 BEAM_DEBUG=1 NV=1 python3 -c "
from tinygrad import Tensor
a = Tensor.randn(256, 256)
(a @ a.T).realize()
"

BEAM_SEARCH:
   0.00s:               from   1 ->   1 actions ...
   0.82s:  12.000 µs      42/  187         (256, 256, <LOCAL 16>, <REDUCE 256>)
   1.94s:   8.200 µs      31/  143         (256, 16, <LOCAL 16>, <UPCAST 4>, <REDUCE 256>)
   2.41s:   8.200 µs       0/   89         (256, 16, <LOCAL 16>, <UPCAST 4>, <REDUCE 256>)
BEAM_SEARCH: final tm=8.200 µs, applied_opts=[LOCAL(0,16), UPCAST(1,4)]
```

Reading the output:
- `from 1 -> 1 actions`: started with 1 kernel, generated 187 candidates
- `42/187`: 42 of 187 compiled and timed successfully
- `12.000 µs → 8.200 µs`: improved from 12µs to 8.2µs
- `0/89`: no further improvement → stop

## BEAM Internals: Filtering

BEAM has several filters to avoid wasting time on bad candidates:

### 1. UOp Count Limit

```python
if len(p.uops) >= BEAM_UOPS_MAX:  # default: 3000
    raise RuntimeError("too many uops")
```

Kernels with thousands of UOps are likely over-unrolled and will be slow.

### 2. Compute Ops Filter

```python
least_compute_ops = min(this_compute_ops, least_compute_ops)
if least_compute_ops * 1000 < this_compute_ops:
    continue  # 1000x more compute than smallest → skip
```

If a candidate has 1000× more arithmetic operations than the best so far, it's probably a bad optimization. Skip timing it.

### 3. Upcast/Local Limits

```python
max_up = getenv("BEAM_UPCAST_MAX", 256)
max_lcl = getenv("BEAM_LOCAL_MAX", 1024)

if up // tc_up > max_up or lcl > max_lcl:
    continue  # too many elements per thread or too many threads per block
```

### 4. Duplicate Library Detection

```python
if lib in seen_libs:
    continue  # exact same binary → already timed
```

Different optimization paths can produce identical compiled code. No need to time it twice.

### 5. Compile Timeout

```python
signal.alarm(getenv("BEAM_TIMEOUT_SEC", 10))  # 10 second timeout per compile
```

Some pathological optimization combinations cause the compiler to hang. Kill it after 10 seconds.

## Why BEAM Matters for LLM Inference

During LLM token generation, the same ~40 kernels run every single token. Even a 10% improvement per kernel compounds:

```
Without BEAM:  40 kernels × average 150 µs = 6 ms/token → 167 tok/s (theoretical)
With BEAM:     40 kernels × average  50 µs = 2 ms/token → 500 tok/s (theoretical)

Orin reality (bandwidth-limited):
Without BEAM heuristic:   ~7.6 tok/s (bad heuristic choices)
With BEAM cache:          ~36.7 tok/s (empirically optimal kernels)
```

The 4.8× improvement comes from:
1. **Better memory access patterns**: UPCAST enables vector loads (128-bit instead of 32-bit)
2. **Correct local group sizes**: matching the SM's warp size and register file
3. **Tensor cores when applicable**: 16× throughput for matmul kernels in prefill
4. **Loop reordering**: SWAP puts the innermost loop on the best memory axis

## BEAM + Heuristic Interaction

When `BEAM=N` is set, the compilation flow becomes:

```
1. heuristic.py: hand_coded_optimizations(s)     ← apply known-good rules
2. search.py: beam_search(s, rawbufs, N)          ← empirically improve further
```

BEAM starts from the heuristic-optimized kernel, not from scratch. The heuristic gives a good starting point (tensor cores, basic LOCAL/UPCAST), and BEAM refines from there.

Without the heuristic's matvec fix ([Pill 11](11-matvec-heuristic.md)), BEAM would start from a terrible baseline and struggle to find the fast path through its limited search budget.

## Summary

- **BEAM search**: bounded BFS over optimization choices, timed on real GPU hardware
- **~190 actions**: UPCAST, UNROLL, LOCAL, GROUP, TC, SWAP, THREAD — covering all kernel transforms
- **Multiprocessing**: compiles candidates in parallel via `fork` on Linux
- **Disk cache**: first run slow (searching), subsequent runs fast (cached results)
- **Filters**: UOp count, compute ops, upcast/local limits, duplicate libs, compile timeout
- **4.8× speedup** on Orin for LLM decode when using cached BEAM results
- **BEAM=2** is usually enough; BEAM=8 gives diminishing returns

---

**Previous**: [← Pill 6: Schedule & Memory](06-schedule-memory.md)
**Next**: [Pill 8: Pattern Matching & Graph Rewriting →](08-pattern-matching.md)
