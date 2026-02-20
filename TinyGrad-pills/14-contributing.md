# Pill 14: Contributing to TinyGrad

## The Philosophy

TinyGrad has a hard rule: **stay under ~10,000 lines**. Every PR that adds lines must justify them. Code that doesn't earn its keep gets deleted.

This means:
- No dead code, no "just in case" abstractions
- Every function is used, every branch is reachable
- If you can do something in fewer lines without sacrificing clarity, do it

The project tracks line count with `sz.py`:
```bash
python3 sz.py
# tinygrad: 9847 lines
```

## Repository Structure

```
tinygrad/
├── tinygrad/               # The framework (THIS is the 10K lines)
│   ├── tensor.py           # User API, backward(), Tensor class
│   ├── dtype.py            # Type system (dtypes, DType)
│   ├── device.py           # Device abstraction, Compiled, Allocator
│   ├── helpers.py          # Utility functions, env var helpers
│   ├── gradient.py         # compute_gradient (autograd core)
│   ├── nn/                 # Neural network layers + optimizers
│   │   ├── __init__.py     # Linear, Conv2d, LayerNorm, Embedding, etc.
│   │   ├── optim.py        # SGD, Adam, AdamW, LARS, LAMB, Muon
│   │   ├── state.py        # save/load (safetensors)
│   │   └── datasets.py     # MNIST, CIFAR
│   ├── uop/                # Intermediate representation
│   │   └── ops.py          # UOp, UPat, PatternMatcher, Ops enum
│   ├── codegen/            # Kernel code generation
│   │   └── opt/
│   │       ├── heuristic.py  # Default optimization heuristic
│   │       └── search.py     # BEAM search
│   ├── engine/             # Execution engine
│   │   ├── schedule.py     # Graph → schedule of kernels
│   │   └── memory.py       # TLSF memory planner
│   ├── renderer/           # PTX, CUDA, Metal, OpenCL renderers
│   ├── runtime/            # Backend implementations
│   │   ├── ops_nv.py       # NVIDIA NV backend (our focus)
│   │   ├── ops_cuda.py     # NVIDIA CUDA backend
│   │   ├── ops_amd.py      # AMD backend
│   │   └── support/
│   │       └── hcq.py      # Hardware Command Queue abstraction
│   └── viz/                # Visualization tools
├── test/                   # Test suite
│   ├── test_ops.py         # Core operations tests
│   ├── test_tiny.py        # Minimal regression tests
│   ├── test_tensor.py      # Tensor API tests
│   ├── test_schedule.py    # Scheduler tests  
│   ├── test_linearizer.py  # Kernel optimization tests
│   └── test_nn.py          # Neural network layer tests
├── extra/                  # Models, examples, utilities (not counted in 10K)
├── docs/                   # Documentation
└── examples/               # Example scripts
```

## Development Setup

### On Orin (NixOS)

If you're using the NixOS flake from this repo:

```bash
cd /home/agent/jetpack-nixos/examples/tinygrad
nix develop
# You're now in a shell with Python, CUDA, and all deps

cd tinygrad
NV=1 python3 -m pytest test/test_ops.py -v --tb=short -x
```

### Generic Setup

```bash
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
pip install -e ".[testing]"

# Run tests on CPU
python3 -m pytest test/test_ops.py -v --tb=short

# Run tests on NV backend
NV=1 python3 -m pytest test/test_ops.py -v --tb=short
```

## Running Tests

### Key Test Files

| Test File | What It Tests | Run Time |
|-----------|--------------|----------|
| `test_tiny.py` | Minimal smoke tests | ~10 sec |
| `test_ops.py` | All tensor operations | ~2 min |
| `test_tensor.py` | Tensor API, shapes | ~1 min |
| `test_schedule.py` | Schedule correctness | ~30 sec |
| `test_linearizer.py` | Kernel optimization | ~1 min |
| `test_nn.py` | NN layers correctness | ~1 min |
| `test_dtype.py` | Data type handling | ~30 sec |

### Running Specific Tests

```bash
# Single test
NV=1 python3 -m pytest test/test_ops.py::TestOps::test_matmul -v

# Pattern match
NV=1 python3 -m pytest test/test_ops.py -k "matmul" -v

# Stop on first failure
NV=1 python3 -m pytest test/test_ops.py -x

# Parallel (faster, but may have GPU memory issues)
NV=1 python3 -m pytest test/test_ops.py -n auto
```

### The Test Strategy

Tests compare TinyGrad's output against **PyTorch** (or NumPy):

```python
# Typical test pattern in test_ops.py
def test_matmul(self):
    x = Tensor.randn(4, 8)
    y = Tensor.randn(8, 6)
    result = (x @ y).numpy()
    expected = x.numpy() @ y.numpy()
    np.testing.assert_allclose(result, expected, atol=1e-6)
```

Every operation has a reference implementation to compare against. If TinyGrad disagrees with PyTorch/NumPy, that's a bug.

## Linting and Type Checking

```bash
# Ruff (fast linter)
ruff check tinygrad/

# Pylint
pylint tinygrad/

# Mypy (type checking)
mypy tinygrad/

# Pre-commit (runs all checks)
pre-commit run --all-files
```

TinyGrad uses type annotations heavily. New code should be fully typed.

## Making Changes

### Typical Workflow

1. **Reproduce the issue** — write a minimal repro script
2. **Add DEBUG flags** — `DEBUG=4 NV=1 python3 repro.py`
3. **Find the relevant code** — it's only 10K lines, grep works great
4. **Make the fix** — smallest change possible
5. **Test** — `python3 -m pytest test/test_ops.py -x`
6. **Check line count** — `python3 sz.py`

### Example: The Matvec Fix

Our [matvec heuristic fix](11-matvec-heuristic.md) was a perfect example:

```
Problem:  7.6 tok/s on Orin (too slow)
Repro:    NV=1 python3 bench_qwen3_beam.py
Debug:    DEBUG=3 showed no GROUP_REDUCE applied
Root cause: CAST wrapping MUL blocked pattern match in _find_indices
Fix:      7 lines changed in heuristic.py
Result:   36.71 tok/s
```

### Areas to Contribute

| Area | Difficulty | Impact |
|------|-----------|--------|
| Fix a failing test on NV | Easy | Medium |
| Improve error messages | Easy | High |
| Optimize a specific kernel | Medium | High |
| Add a new model to examples | Medium | Medium |
| Fix a renderer edge case | Medium | High |
| Improve the heuristic | Hard | Very High |
| Add a new backend | Hard | Very High |

### What Makes a Good PR

1. **Small** — ideally <50 lines changed
2. **Tested** — includes or references passing tests
3. **Motivated** — explains what problem it solves and how
4. **Net-negative or neutral line count** — deletions are celebrated

## Debugging Toolkit

### DEBUG Levels Cheat Sheet

```bash
DEBUG=1  # Timing per schedule batch + cache hits
DEBUG=2  # Per-kernel timing, shapes, memory usage
DEBUG=3  # Applied optimizations per kernel (UPCAST, LOCAL, GROUP)
DEBUG=4  # Generated source code (PTX/CUDA/Metal)
DEBUG=5  # UOp graph before/after each pass
DEBUG=7  # Disassembled binary code
```

### VIZ Mode

```bash
VIZ=1 python3 my_script.py
# Opens a browser with interactive UOp graph visualization
```

### PROFILE Mode

```bash
PROFILE=1 NV=1 python3 my_script.py
# Generates trace file for chrome://tracing
```

### Common Debug Patterns

```python
# Print UOp graph for any tensor
from tinygrad import Tensor
x = Tensor.randn(4, 4)
y = (x + 1).relu()
print(y.uop)  # Shows the lazy computation graph

# Force realization at specific point
intermediate = model.layer1(x)
intermediate.realize()  # Execute up to here

# Check what device is being used
from tinygrad import Device
print(Device.DEFAULT)  # "NV" if NV=1 is set
```

## Code Style

TinyGrad has a distinctive style. Match it:

### Do

```python
# One-liners when clear
def __call__(self, x): return x.linear(self.weight.T, self.bias)

# Compact conditionals
if self.bias is not None: x = x + self.bias

# Walrus operator
if (cached := self._cache.get(key)) is not None: return cached

# List comprehensions over loops
grads = [compute_grad(p) for p in params if p.requires_grad]
```

### Don't

```python
# Don't use verbose multi-line when one line works
def __call__(self, x):
    result = x.linear(self.weight.T, self.bias)
    return result

# Don't add unnecessary type checks
assert isinstance(x, Tensor)  # not needed if typed

# Don't leave commented-out code
# old_result = x.matmul(w)  # DELETE THIS
```

### Naming

- Classes: `PascalCase` (`HWQueue`, `NVDevice`)
- Functions/methods: `snake_case` (`get_kernel_actions`, `beam_search`)
- Constants: `UPPER_CASE` (`DEBUG`, `BEAM`, `NV`)
- Private: `_prefix` (`_find_indices`, `_internal_memory_planner`)

## Understanding the CI

TinyGrad runs tests across multiple backends:

```
CPU → test_ops, test_tensor, test_nn        (reference)
NV  → test_ops, test_linearizer             (NVIDIA userspace)
AMD → test_ops                              (AMD userspace)
GPU → test_ops                              (OpenCL/Metal)
```

If your change passes on CPU but fails on NV, the issue is likely in:
- The renderer (`renderer/ptx.py`)
- The runtime (`runtime/ops_nv.py`)
- The HCQ layer (`runtime/support/hcq.py`)
- A kernel optimization that generates bad code for that backend

## Key Environment Variables

| Variable | Values | Effect |
|----------|--------|--------|
| `NV` | 0/1 | Use NVIDIA userspace driver |
| `DEBUG` | 0-7 | Verbosity level |
| `BEAM` | 0-N | BEAM search width (0=heuristic) |
| `PROFILE` | 0/1 | Generate trace file |
| `VIZ` | 0/1 | Interactive graph visualization |
| `NOOPT` | 0/1 | Disable kernel optimization |
| `FUSE_OPTIM` | 0/1 | Optimizer fusion |
| `CONST_LR` | 0/1 | Constant learning rate tensor |
| `WINO` | 0/1 | Winograd convolution |
| `IMAGE` | 0/1/2 | Image dtype usage |

## Where to Start

1. **Read the code** — start with `tensor.py`, then `device.py`, then `uop/ops.py`
2. **Run the tests** — `NV=1 python3 -m pytest test/test_tiny.py -v` 
3. **Pick a small issue** — GitHub Issues labeled "good first issue"
4. **Use DEBUG** — understand what happens before changing anything
5. **Ask in Discord** — the community is active and helpful

The best way to understand TinyGrad is to trace a simple operation end-to-end:

```bash
# Trace a matmul from Tensor API to GPU execution
DEBUG=4 NV=1 python3 -c "
from tinygrad import Tensor
a = Tensor.randn(4, 4)
b = Tensor.randn(4, 4)
(a @ b).realize()
"
```

This prints the generated PTX — you can see exactly what GPU code your Python produced.

## Summary

- **10K line limit**: every line must earn its keep
- **Test against PyTorch**: `test_ops.py` is the ground truth
- **DEBUG=N is your best friend**: use it before and after every change
- **Small PRs win**: prefer 7-line fixes over 700-line refactors
- **Match the style**: one-liners, walrus operators, comprehensions
- **Run `sz.py`**: check your line count impact
- **Reproduce first, fix second**: always have a repro script

---

**Previous**: [← Pill 13: Machine Learning Primer](13-ml-primer.md)
**Next**: [Pill 15: Why TinyGrad Wins and Loses →](15-why-tinygrad-wins-and-loses.md)
**Index**: [All Pills →](README.md)
