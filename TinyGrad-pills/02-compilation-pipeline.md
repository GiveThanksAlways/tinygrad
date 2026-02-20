# Pill 2: The Compilation Pipeline

## The Big Picture

When you write `z = (x @ y + bias).relu()`, TinyGrad doesn't execute anything. It builds a **lazy UOp graph**. Only when you call `.realize()` or `.numpy()` does the pipeline kick in:

```
Tensor Code → UOp Graph → Schedule → Optimize → Lower → Render → Compile → Execute
```

Each stage transforms the representation, getting closer to metal. This pill walks through every stage.

## Stage 1: Graph Construction (tensor.py)

Every tensor op appends a UOp to the lazy graph:

```python
x = Tensor.randn(1024, 1024)   # UOp(Ops.BUFFER, ...)
y = x * 2                       # UOp(Ops.MUL, src=(x.uop, CONST(2)))
z = y + 1                       # UOp(Ops.ADD, src=(y.uop, CONST(1)))
w = z.sum()                     # UOp(Ops.REDUCE, arg=Ops.ADD, src=(z.uop,))
```

Cost: essentially free. Just Python object construction.

**Key point**: The graph captures the *intent*, not the execution plan. `x * 2 + 1` could be one fused kernel or two — the scheduler decides later.

## Stage 2: Scheduling (engine/schedule.py)

Scheduling does two things:
1. **Groups UOps into kernels** (which ops fuse together?)
2. **Produces ExecItems** (one per kernel launch)

### Fusion Rules

TinyGrad fuses aggressively. The basic rule: **element-wise ops fuse with their consumers**.

```python
y = x * 2       # Fuses with...
z = y + 1       # ...this. One kernel: x * 2 + 1
w = z.sum()     # Reduction breaks the chain. New kernel.
```

Fusion boundaries:
- **Reductions** (`sum`, `max`) — need all elements before producing output
- **Buffer views** — different memory layout forces a copy
- **Multi-output** — some patterns can't fuse (yet)

### ExecItem

Each kernel becomes an `ExecItem`:

```python
@dataclass
class ExecItem:
  ast: UOp         # The kernel's UOp AST (rooted at Ops.SINK)
  bufs: list[Buffer]  # Input and output buffers
  metadata: tuple   # Debugging info
  prg: Runner|None  # Compiled program (populated later)
```

The `ast` is the kernel's computation tree — everything this kernel needs to do.

## Stage 3: Optimization (codegen/opt/)

This is where TinyGrad decides **how** to execute each kernel on the GPU. Two paths:

### Path A: Heuristic (Default)

`hand_coded_optimizations()` in `heuristic.py` applies rules:
- Detect **matvec** patterns → apply GROUP/LOCAL/UPCAST
- Detect **tensor core** opportunities → apply TC opts
- Apply **upcast** for small dimensions
- Apply **local** for expand axes
- Apply **unroll** for small reductions

Fast (microseconds), decent results.

### Path B: BEAM Search

`BEAM=N` tries many optimization schedules on real hardware, keeping the best N at each step. Much slower (seconds to minutes), but finds schedules the heuristic misses.

More on BEAM in [Pill 7](07-beam-search.md).

### What Gets Optimized?

The optimizer manipulates **axis types** — how each dimension of the computation maps to GPU resources:

| Axis Type | GPU Mapping | What It Does |
|-----------|-------------|--------------|
| `GLOBAL` | Grid blocks | Parallelizes across SMs |
| `LOCAL` | Threads in a block | Parallelizes within an SM |
| `REDUCE` | Sequential loop | Accumulates values |
| `UPCAST` | Unrolled in register | Multiple outputs per thread |
| `UNROLL` | Unrolled loop | Removes loop overhead |
| `GROUP` | Cooperative reduction | Splits reduction across threads |

**Example**: A vector add `y[i] = a[i] + b[i]` with 1024 elements might become:
- `GLOBAL=256` (256 blocks)
- `LOCAL=4` (4 threads per block)
- Each thread handles 1 element

Or with UPCAST:
- `GLOBAL=128` (128 blocks)
- `LOCAL=4`, `UPCAST=2`
- Each thread handles 2 elements (less launch overhead)

## Stage 4: Lowering (codegen/__init__.py → full_rewrite_to_sink)

This is the heavy lifting. `full_rewrite_to_sink()` transforms the high-level UOp AST into low-level code UOps through **many** graph rewrite passes:

```python
def full_rewrite_to_sink(sink, ren, optimize=True):
    # 1. Movement ops
    sink = graph_rewrite(sink, pm_mops+pm_syntactic_sugar)

    # 2. Optimize (heuristic or BEAM)
    if optimize: sink = apply_opts(sink, ren)

    # 3. Expand vectorized ops
    sink = graph_rewrite(sink, expander)

    # 4. Add local buffers
    sink = graph_rewrite(sink, pm_add_buffers_local+rangeify_codegen)

    # 5. Remove reductions → sequential code
    sink = graph_rewrite(sink, pm_reduce)

    # 6. Add GPU dimensions (threadIdx, blockIdx)
    sink = graph_rewrite(sink, pm_add_gpudims)

    # 7. Add memory loads
    sink = graph_rewrite(sink, pm_add_loads)

    # 8. Devectorize (SIMD → scalar if needed)
    sink = graph_rewrite(sink, devectorize)

    # 9. Lower index types to concrete ints
    sink = graph_rewrite(sink, pm_lower_index_dtype)

    # 10. Decompose unsupported ops
    sink = graph_rewrite(sink, pm_decomp)

    # 11. Transcendental functions (sin, cos, exp)
    sink = graph_rewrite(sink, pm_transcendental)

    # 12. Final device-specific rewrites
    sink = graph_rewrite(sink, pm_final_rewrite)

    # 13. Add control flow (IF/ENDIF for gated stores)
    sink = graph_rewrite(sink, pm_add_control_flow)

    return sink
```

Each `graph_rewrite` applies a `PatternMatcher` — a set of rules that match UOp patterns and replace them. This is TinyGrad's secret weapon. More in [Pill 8](08-pattern-matching.md).

### What the Lowered Code Looks Like

Before lowering:
```
REDUCE(ADD, MUL(LOAD(buf0, idx), LOAD(buf1, idx)))
```

After lowering:
```
RANGE(0, 1024)          # for i in range(1024)
  LOAD(buf0, i)         #   tmp0 = buf0[i]
  LOAD(buf1, i)         #   tmp1 = buf1[i]
  MUL(tmp0, tmp1)       #   tmp2 = tmp0 * tmp1
  ACC += tmp2           #   acc += tmp2
STORE(buf2, 0, ACC)     # buf2[0] = acc
```

## Stage 5: Linearization

After lowering, the UOp graph is a DAG (directed acyclic graph). GPUs need **linear** instruction sequences. Linearization topologically sorts the UOps into a flat list:

```python
def do_linearize(prg, sink):
    lst = linearize(sink)  # Topological sort
    # Insert IF/ENDIF for gated stores if needed
    lst = line_rewrite(lst, pm_linearize_cleanups)
    return prg.replace(src=prg.src + (UOp(Ops.LINEAR, src=tuple(lst)),))
```

The output is now a list of UOps in execution order — ready for code generation.

## Stage 6: Rendering (renderer/)

Each device has a renderer that converts UOp lists into source code:

| Device | Renderer | Output Language |
|--------|----------|-----------------|
| NV | `PTXRenderer` | NVIDIA PTX assembly |
| CUDA | `CUDARenderer` | CUDA C |
| CPU | `CStyleRenderer` | C |
| METAL | `MetalRenderer` | Metal Shading Language |
| AMD | `AMDRenderer` | RDNA assembly |

### PTX Example

For `y[i] = x[i] * 2 + 1`:

```ptx
.visible .entry test(
    .param .u64 buf0,    // x
    .param .u64 buf1     // y
) {
    .reg .f32 %f<4>;
    .reg .u64 %r<4>;

    // Get thread index
    mov.u32 %r0, %tid.x;

    // Load x[i]
    ld.global.f32 %f0, [%r0*4 + buf0];

    // Compute x[i] * 2 + 1
    mul.f32 %f1, %f0, 0f40000000;     // * 2.0
    add.f32 %f2, %f1, 0f3f800000;     // + 1.0

    // Store y[i]
    st.global.f32 [%r0*4 + buf1], %f2;

    ret;
}
```

### The Renderer Pipeline in Code

```python
pm_to_program = PatternMatcher([
    # Step 1: Linearize the sink
    (UPat(Ops.PROGRAM, src=(UPat(Ops.SINK), UPat(Ops.DEVICE))), do_linearize),
    # Step 2: Render to source code
    (UPat(Ops.PROGRAM, src=(_, _, UPat(Ops.LINEAR))), do_render),
    # Step 3: Compile to binary
    (UPat(Ops.PROGRAM, src=(_, _, _, UPat(Ops.SOURCE))), do_compile),
])
```

Even the compile pipeline itself uses pattern matching!

## Stage 7: Compilation (runtime/support/compiler_cuda.py)

The rendered source code gets compiled to device binary:

| Compiler | Input | Output |
|----------|-------|--------|
| `NVRTCCompiler` | PTX | CUBIN (via nvrtc) |
| `NVCCCompiler` | CUDA C | CUBIN (via nvcc) |
| `ClangCompiler` | C | Shared object |

**Caching**: Compilation results are cached by source hash. The same kernel won't be recompiled twice in a session.

```python
# In compiler_cuda.py
def compile_cached(self, src):
    key = hashlib.sha256(src.encode()).hexdigest()
    if key in self._cache: return self._cache[key]
    lib = self.compile(src)
    self._cache[key] = lib
    return lib
```

## Stage 8: Execution (runtime/ops_nv.py)

The compiled binary gets loaded and launched:

```python
class CompiledRunner(Runner):
    def __call__(self, rawbufs, var_vals, wait=False):
        global_size, local_size = self.p.launch_dims(var_vals)
        return self._prg(
            *[x._buf for x in rawbufs],
            global_size=tuple(global_size),
            local_size=tuple(local_size),
            wait=wait
        )
```

On the NV backend, this translates to:
1. Write kernel arguments to a **constant buffer** (QMD)
2. Write a **pushbuffer** command (COMPUTE_DISPATCH)
3. Write a **GPFIFO** entry pointing to the pushbuffer
4. Ring the **doorbell** (MMIO write at offset 0x90)
5. GPU wakes up, reads GPFIFO, executes kernel

## The Full Pipeline in One Diagram

```
Tensor([1,2,3]) * 2 + 1
         │
    ┌────▼────┐
    │  LAZY   │  Python: UOp graph construction (nanoseconds)
    │  GRAPH  │  MUL(BUFFER, CONST(2)) → ADD(_, CONST(1))
    └────┬────┘
         │ .realize()
    ┌────▼────┐
    │SCHEDULE │  Python: Fuse ops, create ExecItems (microseconds)
    │         │  One ExecItem: x*2+1 fused into single kernel
    └────┬────┘
         │
    ┌────▼────┐
    │OPTIMIZE │  Python: Heuristic or BEAM (μs to minutes)
    │         │  Choose: GLOBAL=N, LOCAL=M, UPCAST=K
    └────┬────┘
         │
    ┌────▼────┐
    │  LOWER  │  Python: 12+ rewrite passes (milliseconds)
    │         │  High-level UOps → low-level UOps
    └────┬────┘
         │
    ┌────▼────┐
    │ RENDER  │  Python: UOps → PTX/C/Metal source (microseconds)
    │         │  Emit device-specific text
    └────┬────┘
         │
    ┌────▼────┐
    │ COMPILE │  C/NVRTC: Source → binary (10-100ms)
    │         │  PTX → CUBIN (cached)
    └────┬────┘
         │
    ┌────▼────┐
    │ EXECUTE │  GPU: Launch kernel (microseconds to milliseconds)
    │         │  Doorbell → GPFIFO → Compute → Signal
    └────┬────┘
         │
       Result
```

## Seeing It Yourself

```bash
# See fused kernels (scheduling)
DEBUG=3 NV=1 python3 -c "from tinygrad import Tensor; (Tensor.randn(100)*2+1).realize()"

# See generated code (rendering)
DEBUG=4 NV=1 python3 -c "from tinygrad import Tensor; (Tensor.randn(100)*2+1).realize()"

# See assembly (compilation)
DEBUG=7 NV=1 python3 -c "from tinygrad import Tensor; (Tensor.randn(100)*2+1).realize()"

# Visualize the graph (all stages)
VIZ=1 python3 -c "from tinygrad import Tensor; (Tensor.randn(100)*2+1).realize()"
```

## Key Insight: Everything is Graph Rewriting

The compilation pipeline is ~15 calls to `graph_rewrite()` with different `PatternMatcher` sets. Each pass:
1. Walks the UOp graph
2. Matches patterns
3. Replaces matched subgraphs
4. Repeats until no more matches

This is fundamentally different from traditional compilers (LLVM, GCC) which use explicit pass managers with hand-coded IR transformations. TinyGrad's approach is **declarative** — you describe what patterns to look for and what to replace them with. The engine handles traversal.

The practical effect: adding a new optimization is **one pattern rule**, not a 500-line compiler pass.

## Method Cache

Compiled runners are cached by AST + context:

```python
def get_runner(device, ast):
    context = (BEAM.value, NOOPT.value, DEVECTORIZE.value, EMULATED_DTYPES.value)
    ckey = (device, type(Device[device].compiler), ast.key, context, False)
    if cret := method_cache.get(ckey): return cret
    # ... compile and cache ...
```

Same kernel with same options → same compiled binary. This is why the JIT is fast on repeated calls.

## Summary

| Stage | Input | Output | Time |
|-------|-------|--------|------|
| Graph Construction | Python ops | UOp DAG | ~ns |
| Scheduling | UOp DAG | ExecItems | ~μs |
| Optimization | ExecItem AST | Optimized AST | μs-min |
| Lowering | Optimized AST | Low-level UOps | ~ms |
| Linearization | UOp DAG | UOp list | ~μs |
| Rendering | UOp list | PTX/C source | ~μs |
| Compilation | Source | Binary | 10-100ms |
| Execution | Binary + buffers | GPU result | μs-ms |

---

**Previous**: [← Pill 1: Tensor & UOp Fundamentals](01-tensor-uop.md)
**Next**: [Pill 3: GPU Architecture Primer →](03-gpu-architecture.md)
