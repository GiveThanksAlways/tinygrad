# Pill 6: Schedule & Memory Management

## From Lazy to Scheduled

In [Pill 1](01-tensor-uop.md) you learned that TinyGrad is lazy — operations build a UOp graph but don't execute anything until `.realize()` is called. In [Pill 2](02-compilation-pipeline.md) you saw the 8-stage pipeline that compiles and runs kernels.

This pill zooms into the two middle steps that bridge lazy computation and actual execution: **scheduling** and **memory planning**.

```
Lazy UOp Graph ──► Schedule ──► Memory Plan ──► Execute
                   (this pill)                  (Pill 5)
```

## What is a Schedule?

A **schedule** is an ordered list of `ExecItem`s — each one represents a GPU kernel to run, plus the buffers it reads and writes:

```python
ExecItem = namedtuple-like(
    ast: UOp,              # the computation (kernel AST)
    bufs: list[Buffer],    # [output_buf, input_buf0, input_buf1, ...]
    metadata: tuple,       # debugging info (operation names, shapes)
    fixedvars: dict,       # bound variable values (e.g., sequence length)
)
```

When you write:

```python
a = Tensor.randn(1024)
b = Tensor.randn(1024)
c = (a * b + 1).relu()
c.realize()
```

The schedule might be:

```
ExecItem 0: kernel=randn,     bufs=[buf_a]           # generate random a
ExecItem 1: kernel=randn,     bufs=[buf_b]           # generate random b
ExecItem 2: kernel=fused_mul_add_relu, bufs=[buf_c, buf_a, buf_b]  # c = relu(a*b+1)
```

Notice: `a*b+1` and `relu` were **fused** into one kernel. That fusion happened during the rangeify/kernelize step. The schedule just captures the final plan.

## The Scheduling Pipeline

`complete_create_schedule_with_vars()` is the main entry point. It takes the big lazy UOp graph and returns a schedule:

```
big_sink (lazy UOp)
    │
    ├── 1. Cache normalization (replace BUFFERs with PARAMs, strip BINDs)
    │
    ├── 2. Schedule cache lookup
    │      hit? → reuse previous schedule    ← huge speedup for LLM decode
    │      miss? → continue below
    │
    ├── 3. Multi-device handling (split across GPUs if needed)
    │
    ├── 4. Rangeify (decide kernel boundaries, fuse operations)
    │
    ├── 5. create_schedule() (topological sort + linearize)
    │
    ├── 6. Unroll outer ranges (handle loops in the schedule)
    │
    ├── 7. Memory planning (reuse buffers to reduce RAM)
    │
    └── Return: (tensor_map, schedule, var_vals)
```

### Step 1: Cache Normalization

Before checking the schedule cache, we normalize the UOp graph so that structurally-identical graphs produce the same cache key — even if their buffers are different objects:

```python
pm_pre_sched_cache = PatternMatcher([
    # BUFFER(UNIQUE, DEVICE) → PARAM(index, ...)  so buffer identity doesn't matter
    (UPat(Ops.BUFFER, ...), replace_input_buffer),
    # BIND(var, value) → BIND(var)  so different values hit same cache
    (UPat(Ops.BIND, ...), strip_bind),
])
```

This means if you run the same model structure with different data, the schedule is reused from cache. Only the buffer pointers and variable values change.

### Step 2: Schedule Cache

```python
schedule_cache: dict[bytes, tuple[list[ExecItem], UOp]] = {}
sched_cache_key = big_sink_cache.key  # hash of the normalized UOp graph

if sched_cache_key in schedule_cache:
    # CACHE HIT — skip all scheduling work
    pre_schedule, combined_sink = schedule_cache[sched_cache_key]
else:
    # CACHE MISS — do the full scheduling pipeline
    ...
```

For LLM inference, this is critical. The decode loop runs the same computation structure every token — only the KV cache contents change. A schedule cache hit means the scheduling step is essentially free after the first token.

You can see cache hits in debug output:
```
scheduled   42 kernels in    0.12 ms |  cache hit a3f7c1d2 | 8421 uops in cache
```

### Step 3: Rangeify (Kernel Boundary Detection)

Rangeify is the pass that decides **what to fuse**. It walks the lazy graph and groups connected operations into kernels:

```
Before rangeify:
  randn_a → mul → add → relu

After rangeify:
  Kernel 0: randn → buf_a
  Kernel 1: load[buf_a] → mul → add → relu → buf_c
```

The rules for fusion:
- **Elementwise ops chain**: `mul → add → relu` are all pointwise → fuse into one kernel
- **Reduce breaks**: a `sum()` or `max()` creates a new kernel boundary
- **Memory breaks**: anything that was already `.realize()`d is a boundary
- **Shape incompatibility**: ops with different shapes can't always fuse

### Step 4: create_schedule() — Topological Sort

Once we know which operations form which kernels, we need to **order** them. A matmul's output must be computed before the relu that consumes it.

```python
def create_schedule(sched_sink: UOp) -> tuple[list[ExecItem], UOp]:
    # Build dependency graph
    children: dict[UOp, list[UOp]] = {}
    in_degree: dict[UOp, int] = {}

    # BFS (Kahn's algorithm) for topological sort
    queue = deque(k for k, v in in_degree.items() if v == 0)
    schedule = []
    while queue:
        k = queue.popleft()
        schedule.append(k)
        for x in children.get(k, []):
            in_degree[x] -= 1
            if in_degree[x] == 0:
                queue.append(x)

    return pre_schedule, buf_uops_sink
```

This is **Kahn's algorithm** — a classic topological sort. It ensures:
1. Kernels with no dependencies run first
2. A kernel only runs after all its inputs are computed
3. Independent kernels can be listed in any order (GPU handles parallelism)

### Step 5: Outer Range Unrolling

Some schedules have outer loops — for example, autoregressive generation where you process one token at a time. The `unroll_outer_ranges` function expands these:

```python
# Before unrolling:
#   RANGE(0, seq_len)
#     KERNEL: attention
#     KERNEL: ffn
#   END

# After unrolling (seq_len=3):
#   ExecItem: attention (fixedvars={i: 0})
#   ExecItem: ffn      (fixedvars={i: 0})
#   ExecItem: attention (fixedvars={i: 1})
#   ExecItem: ffn      (fixedvars={i: 1})
#   ExecItem: attention (fixedvars={i: 2})
#   ExecItem: ffn      (fixedvars={i: 2})
```

Each unrolled iteration gets the loop variable as a `fixedvar`, which is substituted into the kernel's symbolic expressions at launch time.

## Memory Planning

Here's a common problem. Say you have 5 intermediate buffers:

```python
a = randn(1024)          # buf_a: 4 KB
b = randn(1024)          # buf_b: 4 KB
c = a * b                # buf_c: 4 KB (needs buf_a, buf_b)
d = c.relu()             # buf_d: 4 KB (needs buf_c)
e = d.sum()              # buf_e: 4 B  (needs buf_d)
```

Naively, you need 5 buffers = ~16 KB. But after the mul (`c = a * b`), we never read `a` or `b` again. Their memory can be reused!

```
Timeline:    0     1     2     3     4
buf_a:     [alloc]---[free]
buf_b:     [alloc]---[free]
buf_c:              [alloc]---[free]
buf_d:                       [alloc]---[free]
buf_e:                                [alloc]───►

Optimized:
buf_a:     [alloc]───────────────────────────────  ← reused as buf_c, then buf_d
buf_b:     [alloc]---[free]
buf_e:                                [alloc]───►
Total: 8 KB instead of 16 KB
```

### How it Works: TLSF Allocator

TinyGrad uses a **TLSF** (Two-Level Segregated Fit) allocator for memory planning. TLSF is a real-time allocator — O(1) alloc and free, no fragmentation problems.

```python
def _internal_memory_planner(buffers, ...):
    # 1. Find first and last use of each buffer
    for i, kernel_bufs in enumerate(buffers):
        for buf in kernel_bufs:
            first_appearance[buf.base] = min(first_appearance.get(buf.base, i), i)
            last_appearance[buf.base] = i

    # 2. Sort events by timeline position
    buffer_requests = sorted([
        ((first_appearance[buf], True), buf),    # alloc event
        ((last_appearance[buf]+1, False), buf),  # free event
    ])

    # 3. Process events using TLSF allocator
    global_planner = TLSFAllocator(total_memory)
    for (_, is_open), buf in buffer_requests:
        if is_open:
            offset = global_planner.alloc(round_up(buf.nbytes, 0x1000))
            buffer_replace[buf] = (None, offset)
        else:
            global_planner.free(buffer_replace[buf][1])
```

The key insight: instead of allocating/freeing individual Device buffers, the planner allocates one big buffer per device and hands out sub-regions from it:

```
Global buffer (one per device):
┌─────────────────────────────────────────────┐
│  [buf_a region]  [buf_c region]  [buf_e]    │
│  offset=0        offset=4096    offset=8192 │
└─────────────────────────────────────────────┘

buf_a and buf_c overlap in time → can share the same offset
```

### Memory Savings in Practice

```bash
DEBUG=1 NV=1 python3 -c "
from tinygrad import Tensor
x = Tensor.randn(256, 256)
y = (x @ x.T).relu().sum()
y.realize()
"
# memory reduced from 2.62 MB -> 0.52 MB, 10 -> 3 bufs
```

For LLM inference with a 1B parameter model, memory planning can reduce intermediate buffer usage by 50-80%.

### What Gets Excluded

Not all buffers can be reused:

```python
# Skip if:
buf.is_allocated()       # already has backing memory
buf.base.is_allocated()  # base buffer already allocated
buf.uop_refcount > 0     # still referenced by lazy graph
buf in noopt_buffers      # involved in copy/transfer ops (need parallelism)
```

Copy/transfer buffers are excluded because the memory planner assumes sequential execution within a device. If a buffer is being DMA'd while another kernel writes to the same memory, that's a race condition.

## The ExecItem Lifecycle

Putting it all together:

```
1. Tensor operations build UOp graph      (lazy, no execution)
2. .realize() triggers scheduling          (engine/schedule.py)
3. Schedule cache check                    (instant if hit)
4. Topological sort produces ExecItem list (ordered by dependencies)
5. Memory planner assigns buffers          (reuses dead allocations)
6. run_schedule() iterates ExecItems       (engine/realize.py)
7. Each ExecItem → get_runner → compile → launch kernel
8. GPU executes, timeline advances
```

### Debugging the Schedule

```bash
# See schedule creation time and cache status
DEBUG=1 NV=1 python3 my_model.py
# scheduled   42 kernels in    0.12 ms |  cache hit a3f7c1d2

# See each kernel's shape and operation
DEBUG=2 NV=1 python3 my_model.py
# 0.012 ms  matmul(256,256,256)  mem:0.75 MB
# 0.003 ms  relu(256,256)         mem:0.25 MB

# See the full AST for each kernel
DEBUG=5 NV=1 python3 my_model.py
```

## How This Connects to LLM Inference

For a Qwen3 1B decode step:

```
Schedule: ~40 kernels per token
├── Attention: QKV projection (3 matvecs)
├── Attention: RoPE (element-wise)
├── Attention: Softmax (reduce + element-wise)
├── Attention: Value multiply (matvec)
├── FFN: Gate + Up projection (2 matvecs)
├── FFN: SiLU activation (element-wise)
├── FFN: Down projection (matvec)
└── LayerNorm, residual adds, etc.

Schedule cache: HIT (same structure every token)
Memory plan: ~80% reduction in intermediate buffers
Compile cache: all 40 kernels already compiled from first token
```

The first token is slow (cache misses everywhere). Every subsequent token is fast: schedule from cache, kernels from cache, memory plan from cache. This is why TinyGrad's "Time To First Token" (TTFT) is higher than steady-state token generation.

## Summary

- **Scheduling** converts a lazy UOp graph into an ordered list of `ExecItem`s
- **Kahn's algorithm** (topological sort) ensures dependency ordering
- **Schedule cache** makes repeated computations (LLM decode) nearly free to schedule
- **Cache normalization** strips buffer identity and variable values so structurally-identical graphs share schedules
- **Memory planner** uses TLSF allocation to reuse dead buffers, reducing memory 50-80%
- **Outer range unrolling** handles loops in the schedule (autoregressive generation)
- The first execution is slow (all caches cold); subsequent ones are fast

---

**Previous**: [← Pill 5: Device Backends & HCQ](05-device-backends.md)
**Next**: [Pill 7: BEAM Search →](07-beam-search.md)
