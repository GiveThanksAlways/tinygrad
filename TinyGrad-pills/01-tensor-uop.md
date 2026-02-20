# Pill 1: Tensor & UOp Fundamentals

## The Two Core Abstractions

TinyGrad has two fundamental concepts:
1. **Tensor** - The user-facing API (what you write)
2. **UOp** - The internal representation (what the compiler sees)

Everything else builds on these.

## Tensors: The Frontend

### What is a Tensor?

A **multidimensional array** with:
- **Data**: Numbers (floats, ints, etc.)
- **Shape**: Dimensions (e.g., `(64, 128, 3)`)
- **Operations**: Add, multiply, matmul, etc.
- **Autograd**: Automatic differentiation

### Creating Tensors

```python
from tinygrad import Tensor

# From Python list
x = Tensor([1, 2, 3, 4])

# Zeros, ones
z = Tensor.zeros(10, 10)
o = Tensor.ones(3, 3)

# Random
r = Tensor.randn(100, 100)  # Normal distribution
u = Tensor.rand(100, 100)   # Uniform [0, 1)

# Special matrices
eye = Tensor.eye(5)  # Identity matrix
arange = Tensor.arange(0, 100, 5)  # [0, 5, 10, ..., 95]

# With gradient tracking
x = Tensor.randn(10, requires_grad=True)
```

### Tensor Attributes

```python
x = Tensor.randn(64, 128, 3)

x.shape    # (64, 128, 3)
x.dtype    # dtypes.float32
x.device   # "CUDA", "CPU", "NV", etc.
x.requires_grad  # True/False
x.grad     # Gradient (after backward())
```

### Basic Operations

```python
a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])

# Element-wise
c = a + b         # [5, 7, 9]
c = a * b         # [4, 10, 18]
c = a - b         # [-3, -3, -3]
c = a / b         # [0.25, 0.4, 0.5]

# Unary
c = a.sqrt()      # [1, 1.414, 1.732]
c = a.exp()       # [e^1, e^2, e^3]
c = a.log()       # [0, 0.693, 1.099]
c = -a            # [-1, -2, -3]

# Activation functions
c = a.relu()      # max(0, a)
c = a.sigmoid()   # 1 / (1 + e^(-a))
```

### Shape Operations

```python
x = Tensor.randn(2, 3, 4)

# Reshape
y = x.reshape(6, 4)        # (6, 4)
y = x.reshape(-1, 4)       # Infer first dim

# Transpose
y = x.T                     # (4, 3, 2)
y = x.transpose(1, 2)       # Swap axes 1 and 2

# Permute
y = x.permute(2, 0, 1)     # (4, 2, 3)

# Expand (broadcast)
x = Tensor.ones(1, 3)
y = x.expand(5, 3)          # (5, 3) - repeats row

# Pad
y = x.pad(((0, 1), (2, 2)))  # Add padding

# Squeeze/unsqueeze
x = Tensor.ones(1, 3, 1, 4)
y = x.squeeze()             # (3, 4)
y = x.unsqueeze(0)          # (1, 1, 3, 1, 4)
```

### Reduction Operations

```python
x = Tensor.randn(10, 20, 30)

# Sum
y = x.sum()                 # Scalar
y = x.sum(axis=0)           # (20, 30)
y = x.sum(axis=(0, 2))      # (20,)
y = x.sum(keepdim=True)     # (1, 1, 1)

# Other reductions
y = x.mean()                # Average
y = x.max()                 # Maximum
y = x.min()                 # Minimum
```

### Matrix Operations

```python
a = Tensor.randn(64, 128)
b = Tensor.randn(128, 256)

# Matrix multiplication
c = a @ b          # (64, 256)
c = a.matmul(b)    # Same

# Batch matrix multiplication
a = Tensor.randn(10, 64, 128)
b = Tensor.randn(10, 128, 256)
c = a @ b          # (10, 64, 256)
```

### Indexing and Slicing

```python
x = Tensor.randn(10, 20)

# Slicing
y = x[0]           # First row: (20,)
y = x[:, 0]        # First column: (10,)
y = x[2:5]         # Rows 2-4: (3, 20)
y = x[:, ::2]      # Every other column

# Advanced indexing
y = x[x > 0]       # Boolean masking (flattened)
```

## Laziness: The Key to Performance

### Immediate vs Lazy Execution

**Eager (PyTorch)**:
```python
x = torch.randn(1000, 1000)
y = x * 2          # Launches GPU kernel
z = y + 1          # Launches another kernel
```
Two kernel launches = overhead + missed fusion opportunities.

**Lazy (TinyGrad)**:
```python
x = Tensor.randn(1000, 1000)
y = x * 2          # No kernel yet
z = y + 1          # Still no kernel
result = z.numpy() # ONE fused kernel: x * 2 + 1
```

### Forcing Execution

```python
x = Tensor.randn(100, 100)
y = (x * 2 + 1).relu()

# Force execution
y.realize()  # Compile and run

# Get numpy array (also forces execution)
arr = y.numpy()

# Get scalar (also forces execution)
val = y.item()
```

### Why Laziness Matters

1. **Fusion**: Multiple ops → one kernel
2. **Rewriting**: Optimize before execution
3. **Scheduling**: Smarter device usage
4. **Memory**: Avoid intermediate allocations

## UOps: The Internal Representation

### What is a UOp?

A **Universal Operation** - the IR that represents everything:
- Tensor operations (`ADD`, `MUL`)
- Memory access (`LOAD`, `STORE`)
- Control flow (`IF`, `RANGE`)
- Special ops (`REDUCE`, `CAST`)

### UOp Structure

```python
@dataclass(frozen=True, eq=False)
class UOp:
  op: Ops                    # Operation type (ADD, MUL, LOAD, etc.)
  dtype: DType               # Data type (float32, int32, etc.)
  src: tuple[UOp, ...]       # Source operands
  arg: Any = None            # Optional argument
```

**Key properties**:
- **Immutable**: Once created, never changes
- **Cached**: Same UOp created twice → same object
- **DAG**: Directed Acyclic Graph (no cycles)

### Ops Enum

Core operation types:

```python
class Ops(Enum):
  # Unary ops
  NEG = auto()      # -x
  EXP2 = auto()     # 2^x
  LOG2 = auto()     # log2(x)
  SIN = auto()      # sin(x)
  SQRT = auto()     # sqrt(x)
  
  # Binary ops
  ADD = auto()      # x + y
  MUL = auto()      # x * y
  IDIV = auto()     # x // y
  MAX = auto()      # max(x, y)
  MOD = auto()      # x % y
  
  # Ternary ops
  WHERE = auto()    # if c then x else y
  
  # Reduce ops
  REDUCE = auto()   # sum/max over axis
  
  # Movement ops
  EXPAND = auto()   # Broadcast
  PERMUTE = auto()  # Transpose
  PAD = auto()      # Add padding
  SHRINK = auto()   # Slice
  
  # Memory ops
  LOAD = auto()     # Load from buffer
  STORE = auto()    # Store to buffer
  CONST = auto()    # Constant value
  
  # Control flow
  RANGE = auto()    # Loop iterator
  IF = auto()       # Conditional
  BARRIER = auto()  # Synchronization
  
  # Special
  CAST = auto()     # Type conversion
  BITCAST = auto()  # Reinterpret bits
  VECTORIZE = auto() # SIMD vectorization
  
  # ... and more
```

### UOp Examples

#### Simple Math

```python
# Python: z = x + y
x_uop = UOp(Ops.BUFFER, ...)
y_uop = UOp(Ops.BUFFER, ...)
z_uop = UOp(Ops.ADD, dtypes.float32, (x_uop, y_uop))
```

#### Constants

```python
# Python: x = 5
x_uop = UOp(Ops.CONST, dtypes.int32, arg=5)

# Python: y = 3.14
y_uop = UOp(Ops.CONST, dtypes.float32, arg=3.14)
```

#### Memory Load

```python
# Python: x = buffer[index]
buffer = UOp(Ops.BUFFER, ...)
index = UOp(Ops.CONST, dtypes.int32, arg=0)
x = UOp(Ops.LOAD, dtypes.float32, (buffer, index))
```

#### Conditionals

```python
# Python: result = x if condition else y
condition = UOp(Ops.CMPLT, dtypes.bool, (a, b))  # a < b
result = UOp(Ops.WHERE, dtypes.float32, (condition, x, y))
```

### Accessing UOps from Tensors

```python
x = Tensor([1, 2, 3])
y = x + 1

# Get the UOp
print(y.uop)
# UOp(Ops.ADD, dtypes.int32, (
#   UOp(Ops.BUFFER, ...),
#   UOp(Ops.CONST, arg=1)
# ))

# Traverse the graph
for uop in y.uop.toposort():
  print(uop.op, uop.dtype)
```

### Why UOps?

**Uniform IR** = uniform optimization:
- Pattern matching works everywhere
- One set of rewrite rules
- Easy to add new operations
- Compiler stays simple

**Alternative (PyTorch)**:
- Separate IRs for frontend, autograd, JIT, backend
- Different optimization passes for each
- Complex and hard to modify

## UOp Construction and Caching

### The ucache

UOps are **cached by content**:

```python
# Create UOp
x = UOp(Ops.ADD, dtypes.float32, (a, b))

# Create same UOp again
y = UOp(Ops.ADD, dtypes.float32, (a, b))

# They're the SAME object
assert x is y  # True!
```

**Why?** Avoid duplicate nodes in the graph. Saves memory, enables equality checks.

### UOp Methods

```python
uop = UOp(Ops.ADD, dtypes.float32, (x, y))

# Replace fields (creates new UOp)
new_uop = uop.replace(op=Ops.MUL)
new_uop = uop.replace(src=(x, z))

# Traversal
uop.toposort()     # List in topological order
uop.src            # Direct children
uop.parents        # Direct parents

# Properties
uop.dtype          # Data type
uop.op             # Operation
uop.arg            # Argument (if any)
```

### UOp Invariants

Rules that all UOps follow:

1. **Immutability**: Never modify a UOp
2. **Caching**: Same content → same object
3. **Acyclic**: No cycles in the graph
4. **Type safety**: Operations have valid types

## Tensor → UOp Translation

### What Happens When You Create a Tensor Op?

```python
x = Tensor([1, 2, 3])
y = x + 1
```

**Under the hood**:
1. `x` creates a `UOp(Ops.BUFFER, ...)`
2. `1` creates a `UOp(Ops.CONST, arg=1)`
3. `+` creates a `UOp(Ops.ADD, src=(x.uop, const_uop))`
4. `y.uop` holds this graph

No execution happens. Just graph construction.

### Tensor Methods → UOp Translation

| Tensor Operation | UOp Translation |
|------------------|-----------------|
| `x + y` | `UOp(Ops.ADD, src=(x, y))` |
| `x * y` | `UOp(Ops.MUL, src=(x, y))` |
| `x.relu()` | `UOp(Ops.MAX, src=(x, 0))` |
| `x.sum()` | `UOp(Ops.REDUCE, arg=Ops.ADD)` |
| `x.reshape(...)` | `UOp(Ops.RESHAPE, arg=new_shape)` |
| `x[0]` | `UOp(Ops.SHRINK, arg=slice(0, 1))` |

### Complex Example

```python
x = Tensor.randn(10, 10)
y = (x * 2 + 1).sum()
```

**UOp graph**:
```
REDUCE(axis=None, op=ADD)
  └─ ADD
      ├─ MUL
      │   ├─ BUFFER (x)
      │   └─ CONST(2)
      └─ CONST(1)
```

## Autograd with UOps

### Backward Pass

```python
x = Tensor([1, 2, 3], requires_grad=True)
y = (x * 2).sum()
y.backward()
print(x.grad)  # Tensor([2, 2, 2])
```

**What happens**:
1. Forward: Build UOp graph for `y`
2. Backward: Create gradient UOps using chain rule
3. Realize: Execute both forward and gradient graphs

### Gradient UOps

TinyGrad implements autograd by **building a gradient UOp graph**:

```python
# Forward
y = x * 2

# Backward (simplified)
grad_x = grad_y * 2  # dy/dx = 2
```

Each operation has a gradient rule:
- `ADD`: Split gradient
- `MUL`: Scale by other input
- `REDUCE`: Expand gradient
- etc.

## DType: The Type System

### Supported Types

```python
from tinygrad.dtype import dtypes

# Floats
dtypes.float16   # Half precision
dtypes.float32   # Single precision (default)
dtypes.float64   # Double precision

# Integers
dtypes.int8      # 8-bit signed
dtypes.int16     # 16-bit signed
dtypes.int32     # 32-bit signed
dtypes.int64     # 64-bit signed

# Unsigned
dtypes.uint8     # 8-bit unsigned
dtypes.uint16    # 16-bit unsigned

# Special
dtypes.bool      # Boolean
```

### Type Casting

```python
x = Tensor([1.5, 2.7, 3.2])

# Cast to int
y = x.cast(dtypes.int32)  # [1, 2, 3]

# Cast to half precision
y = x.cast(dtypes.float16)
```

### Automatic Type Promotion

```python
x = Tensor([1, 2, 3])          # int32
y = Tensor([1.0, 2.0, 3.0])    # float32

z = x + y  # Automatically promotes to float32
```

## Memory Layout and Strides

### Contiguous vs Strided

Tensors have a **memory layout** defined by strides:

```python
x = Tensor.randn(3, 4)  # Shape (3, 4)
# Memory: [row0, row1, row2] in linear memory
# Strides: (4, 1) - skip 4 elements for next row

y = x.T  # Shape (4, 3)
# Memory: SAME as x (no copy!)
# Strides: (1, 4) - different interpretation
```

### Contiguous Memory

```python
x = Tensor.randn(10, 10)
y = x.T  # Not contiguous

# Force contiguous layout
z = y.contiguous()  # Now contiguous

# Check
y.is_contiguous()  # False
z.is_contiguous()  # True
```

## Practical Examples

### Example 1: Linear Layer

```python
def linear(x, weight, bias):
  return x @ weight + bias

x = Tensor.randn(64, 128)
weight = Tensor.randn(128, 256)
bias = Tensor.randn(256)

y = linear(x, weight, bias)  # (64, 256)
```

### Example 2: ReLU Activation

```python
def relu(x):
  return x.relu()  # Equivalent to x.maximum(0)

x = Tensor([-1, 2, -3, 4])
y = relu(x)  # [0, 2, 0, 4]
```

### Example 3: Softmax

```python
def softmax(x):
  exp_x = (x - x.max(axis=-1, keepdim=True)).exp()
  return exp_x / exp_x.sum(axis=-1, keepdim=True)

x = Tensor.randn(10, 100)
y = softmax(x)  # (10, 100), each row sums to 1
```

### Example 4: Training Loop

```python
from tinygrad import Tensor, nn

# Model
class Linear:
  def __init__(self, in_dim, out_dim):
    self.weight = Tensor.randn(in_dim, out_dim, requires_grad=True)
    self.bias = Tensor.randn(out_dim, requires_grad=True)
  
  def __call__(self, x):
    return x @ self.weight + self.bias

model = Linear(128, 10)
opt = nn.optim.SGD([model.weight, model.bias], lr=0.01)

# Training
for i in range(100):
  # Forward
  x = Tensor.randn(32, 128)  # Batch
  y_true = Tensor.randint(32, low=0, high=10)  # Labels
  
  y_pred = model(x)
  loss = y_pred.sparse_categorical_crossentropy(y_true)
  
  # Backward
  opt.zero_grad()
  loss.backward()
  opt.step()
  
  print(f"Step {i}, Loss: {loss.item()}")
```

## Debugging Tips

### Inspect UOps

```python
x = Tensor([1, 2, 3])
y = (x * 2 + 1).sum()

# Print UOp graph
print(y.uop)

# Print nicely formatted
from tinygrad.helpers import colored
for uop in y.uop.toposort():
  print(f"{uop.op:20s} {uop.dtype}")
```

### Visualize Graphs

```bash
VIZ=1 python3 your_script.py
```

Opens browser with interactive graph visualization.

### Check Shape and Type

```python
x = Tensor.randn(10, 20)
print(f"Shape: {x.shape}")
print(f"DType: {x.dtype}")
print(f"Device: {x.device}")
```

### Force Realization

```python
x = Tensor.randn(100, 100)
y = (x * 2).realize()  # Force execution here

# Now y is "real" (not lazy)
```

## Common Pitfalls

### Pitfall 1: Forgetting to Realize

```python
x = Tensor.randn(1000, 1000)
y = (x * 2 + 1).sum()

# y is still lazy!
# If you time it: 0.0001s (just graph construction)

y.realize()  # NOW it executes
```

### Pitfall 2: Not Checking Shapes

```python
a = Tensor.randn(64, 128)
b = Tensor.randn(256, 128)  # Wrong shape!

c = a @ b  # ERROR: matmul dimension mismatch
```

### Pitfall 3: Modifying Tensors

```python
x = Tensor([1, 2, 3])
x[0] = 5  # ERROR: Tensors are immutable

# Instead:
x = Tensor([5, 2, 3])
```

## Summary

### Key Takeaways

1. **Tensor** = User API, lazy execution, autograd
2. **UOp** = Internal IR, uniform representation
3. **Laziness** = Fusion, optimization, performance
4. **UOp caching** = Memory efficiency, fast equality
5. **Pattern matching** = How the compiler optimizes

### The Big Picture

```
User Code:  z = (x + y) * 2
   ↓
Tensor API: Creates lazy graph
   ↓
UOp Graph:  MUL(ADD(x, y), CONST(2))
   ↓
Compiler:   Optimizes, schedules, generates code
   ↓
Device:     Executes on GPU/CPU
```

## Next Steps

Continue to **[Pill 2: The Compilation Pipeline](02-compilation-pipeline.md)** to see how UOps become runnable GPU kernels.

---

**Previous**: [← Pill 0: Introduction & Philosophy](00-introduction.md)  
**Next**: [Pill 2: The Compilation Pipeline →](02-compilation-pipeline.md)
