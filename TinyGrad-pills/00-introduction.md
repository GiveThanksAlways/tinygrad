# Pill 0: Introduction & Philosophy

## What is TinyGrad?

TinyGrad is an **end-to-end deep learning framework** that combines:
- A PyTorch-like tensor library with autograd
- A complete compiler stack (IR, optimizer, codegen)
- Native backends for NVIDIA, AMD, Metal, CPU, and more
- Training infrastructure (nn, optim, datasets)

**The twist**: The entire system is ~10,000 lines of hackable Python code.

## Why TinyGrad Exists

### The Problem with Existing Frameworks

**PyTorch**: Great API, but the compiler is a black box. Good luck understanding torch.compile or CUDA kernels.

**JAX**: Beautiful functional design, but XLA is inscrutable. Try debugging a JAXPR → HLO → PTX compilation.

**TVM**: Powerful compiler, but it's "just" the compiler. You still need a frontend framework.

### The TinyGrad Solution

**One unified stack** where you can:
1. Write high-level tensor code (`a @ b + c`)
2. See the exact IR it generates (UOps)
3. Watch the compiler optimize it (pattern matching)
4. Inspect the generated GPU code (PTX/C)
5. Modify any part of the pipeline

It's PyTorch's API + JAX's IR-based compilation + TVM's optimization, all visible and hackable.

## Core Philosophy

### 1. Every Line Must Earn Its Keep

No "just in case" code. No feature bloat. If 10 lines can replace 100, use 10.

**Example**: PyTorch's autograd engine is ~50,000 lines. TinyGrad's is ~500.

### 2. Readability Over Cleverness

Code golf is banned. The goal is **minimizing cognitive complexity**, not character count.

**Bad**:
```python
return x if x.op==Ops.CONST else x.replace(src=tuple(s.simplify() for s in x.src))
```

**Good**:
```python
if x.op is Ops.CONST: return x
return x.replace(src=tuple(s.simplify() for s in x.src))
```

### 3. Hackability > Performance

TinyGrad is fast (90% of PyTorch speed), but would sacrifice 10% performance to keep the code 50% simpler.

**Why?** Because:
- Simpler code = easier to optimize later
- Readable code = more contributors
- Visible pipeline = educational value

### 4. Laziness is Powerful

Operations don't execute immediately—they build a computation graph. This enables:
- **Fusion**: `(a * b).sum()` becomes one kernel, not two
- **Rewriting**: `x + 0` → `x` before reaching the GPU
- **Scheduling**: Optimize across operation boundaries

## Architecture Overview (30,000 Feet)

```
User writes:     result = (A @ B + C).relu()

1. TENSOR LAYER (tensor.py)
   Creates lazy computation graph as UOps
   
2. SCHEDULE (engine/schedule.py)
   Converts graph into ExecItems (one per kernel)
   
3. LOWERING (codegen/)
   Optimizes UOps with pattern matching
   - Fuse operations
   - Simplify math
   - Choose optimal config (BEAM search)
   
4. CODEGEN (renderer/)
   Emits device-specific code (PTX for NVIDIA, C for CPU, etc.)
   
5. COMPILE (runtime/)
   Invokes device compiler (nvcc, clang, etc.)
   Caches result
   
6. EXECUTE (runtime/)
   Upload data to GPU
   Launch kernel
   Download result
```

## Key Concepts

### UOp (Universal Operation)

The **one IR** for everything. Tensor operations, kernel code, memory loads—all UOps.

```python
# Tensor UOps (high-level)
UOp(Ops.ADD, src=(x_uop, y_uop))  # x + y

# Kernel UOps (low-level)
UOp(Ops.LOAD, src=(buffer, index))  # buffer[index]
UOp(Ops.RANGE, arg=(0, 1024))       # for i in range(1024)
```

**Why one IR?** Uniform optimization infrastructure. Pattern matchers work everywhere.

### Pattern Matching

The compiler's secret weapon. Transform graphs with rewrite rules:

```python
# Rule: x + 0 → x
UPat(Ops.ADD, src=(UPat.var("x"), UPat.cvar("c"))) → lambda x, c: x if c == 0 else None

# Rule: x * 2 → x << 1
UPat(Ops.MUL, src=(UPat.var("x"), UPat.const(2))) → lambda x: x << 1
```

### Laziness

Nothing executes until you call `.realize()` or `.numpy()`:

```python
x = Tensor([1, 2, 3])
y = x * 2          # No computation yet
z = y + 1          # Still no computation
result = z.numpy() # NOW it executes (fused into one kernel)
```

## The TinyGrad Stack in Detail

### Layer 1: Tensor API (`tensor.py`)

PyTorch-compatible interface:
```python
from tinygrad import Tensor

x = Tensor.randn(1024, 1024, requires_grad=True)
y = Tensor.randn(1024, 1024, requires_grad=True)
z = (x @ y).sum()
z.backward()
print(x.grad)  # Gradients computed
```

### Layer 2: UOp IR (`uop/`)

Immutable computation graph:
```python
x = Tensor([1, 2, 3])
y = x + 1

# y.uop is a UOp graph:
# UOp(Ops.ADD, src=(
#   UOp(Ops.BUFFER, ...),  # x
#   UOp(Ops.CONST, arg=1)  # 1
# ))
```

### Layer 3: Scheduler (`engine/schedule.py`)

Converts UOp graph → ExecItem list:
```python
# Input: y = (a @ b) + c
# Output: [
#   ExecItem(ast=matmul_uop, bufs=[a, b, temp]),
#   ExecItem(ast=add_uop, bufs=[temp, c, y])
# ]
```

### Layer 4: Compiler (`codegen/`, `renderer/`)

ExecItem → optimized device code:
- Pattern match to fuse/simplify
- BEAM search for best config
- Emit PTX/C/Metal code
- Invoke compiler

### Layer 5: Runtime (`runtime/`)

Device-specific execution:
- Memory management
- Kernel launching
- Synchronization
- Profiling

## Comparison to Other Frameworks

| Feature | TinyGrad | PyTorch | JAX | TVM |
|---------|----------|---------|-----|-----|
| Frontend API | ✅ | ✅ | ✅ | ❌ |
| Visible IR | ✅ | ❌ | 🟡 | ✅ |
| Hackable Compiler | ✅ | ❌ | ❌ | 🟡 |
| Performance | 90% | 100% | 95% | 100% |
| Lines of Code | 10k | 1M+ | 500k | 300k |
| Learning Curve | Medium | Easy | Hard | Hard |

## Who Should Use TinyGrad?

### ✅ Great For:
- **Learning**: Understand ML systems from first principles
- **Research**: Rapid experimentation with compiler ideas
- **Embedded**: Deploy minimal runtime to constrained devices
- **Education**: Teaching modern ML frameworks

### ❌ Not Great For:
- **Production at Scale**: PyTorch/JAX are more battle-tested
- **Immediate Performance**: PyTorch is 10% faster (for now)
- **Stable API**: TinyGrad moves fast, breaks things

## Why These Pills Exist

After reading all pills, you will:

1. **Understand TinyGrad** from first principles
   - How tensors become GPU kernels
   - How the compiler optimizes
   - How devices execute code

2. **Understand GPUs**
   - Architecture (warps, SMs, memory hierarchy)
   - Programming models (CUDA, PTX)
   - Performance (bandwidth, compute, occupancy)

3. **Understand ML Systems**
   - Why autograd works
   - How neural networks train
   - What makes inference fast

4. **Contribute Meaningfully**
   - Fix bugs
   - Add features
   - Optimize performance
   - Support new devices

## Getting Started

### Installation

```bash
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
python3 -m pip install -e .
```

### First Program

```python
from tinygrad import Tensor

# Create tensors
x = Tensor.eye(3, requires_grad=True)
y = Tensor([[2.0, 0, -2.0]], requires_grad=True)

# Compute
z = y.matmul(x).sum()

# Backprop
z.backward()

# Results
print(x.grad.tolist())  # [[2, 0, -2], [2, 0, -2], [2, 0, -2]]
print(y.grad.tolist())  # [[0, 0, 0]]
```

### See What TinyGrad Does

```bash
# Show the fused kernel
DEBUG=3 python3 your_script.py

# Show generated code
DEBUG=4 python3 your_script.py

# Visualize the computation graph
VIZ=1 python3 your_script.py
```

## The Philosophy in Practice

### Example: Adding Two Tensors

**PyTorch**:
```python
z = x + y  # Magic happens in C++/CUDA
```
You can't see what happens. It's fast, but opaque.

**TinyGrad**:
```python
z = x + y

# Want to see the UOp?
print(z.uop)

# Want to see the kernel?
DEBUG=4 python3 script.py

# Want to modify the compiler?
# Edit codegen/__init__.py and add a pattern matcher!
```

### Example: Custom Optimization

Want to optimize `x * 2` to use left shift?

**PyTorch**: Good luck diving into the C++ codebase.

**TinyGrad**: Add one line to a pattern matcher:

```python
# In codegen/__init__.py
pm_simplify = PatternMatcher([
  # ... existing patterns ...
  (UPat(Ops.MUL, src=(UPat.var("x"), UPat.const(2))),
   lambda x: UOp(Ops.SHL, x.dtype, (x, UOp.const(x.dtype, 1)))),
])
```

Done. Runs on every graph automatically.

## What Makes TinyGrad Different

### It's Educational

The code is **meant to be read**. Comments explain the "why", not just the "what".

### It's Unified

One IR (UOps) for everything. Not separate IRs for:
- Frontend graphs (like PyTorch)
- Backend graphs (like XLA)
- Kernel IR (like Triton)

### It's Hackable

Clone it. Change it. Break it. Learn from it. That's the point.

### It's Fast Enough

90% of PyTorch's performance with 1% of the code. That's a great trade-off for learning and experimentation.

## Common Misconceptions

### "TinyGrad is Slow"

No. It's ~90% of PyTorch on typical workloads. On some workloads (fused ops), it's faster.

### "TinyGrad is Incomplete"

It has: autograd, nn layers, optimizers, multi-GPU, quantization, symbolic shapes, JIT, and runs real models (LLaMA, ResNet, etc.).

### "TinyGrad is Toy Code"

It's production-quality code that's intentionally simple. Simple ≠ toy.

### "I Need to Be an Expert"

No. If you can read Python and have CS fundamentals, you can learn TinyGrad. These pills are here to help.

## Next Steps

Continue to **[Pill 1: Tensor & UOp Fundamentals](01-tensor-uop.md)** to dive into the core abstractions.

Or jump to:
- **[Pill 3: GPU Architecture Primer](03-gpu-architecture.md)** if you want GPU background first
- **[Pill 9: Jetson AGX Orin](09-jetson-orin.md)** for device-specific information

## Further Reading

- [TinyGrad README](https://github.com/tinygrad/tinygrad)
- [Documentation](https://docs.tinygrad.org/)
- [Discord](https://discord.gg/ZjZadyC7PK)

---

**Next**: [Pill 1: Tensor & UOp Fundamentals →](01-tensor-uop.md)
