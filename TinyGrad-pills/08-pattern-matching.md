# Pill 8: Pattern Matching & Graph Rewriting

## The Compiler's Building Blocks

TinyGrad's compiler is almost entirely built from **pattern matchers**. Instead of hand-written `if/else` chains walking the IR tree, TinyGrad declares rules like:

```python
# "If you see x + 0, replace it with x"
(UPat(Ops.ADD, src=[UPat.var("x"), UPat.const(None, 0)]), lambda x: x)
```

Every optimization pass, every lowering step, every simplification — they're all lists of `(pattern, replacement)` rules applied by `graph_rewrite()`.

This pill covers the three core abstractions: **UPat** (patterns), **PatternMatcher** (rule sets), and **graph_rewrite** (the engine).

## UPat: Pattern Templates

A `UPat` describes what a UOp should look like — without specifying exact values. It's a template with wildcards.

### Basic Matching

```python
# Match any ADD operation
UPat(Ops.ADD)

# Match ADD of two specific sub-patterns
UPat(Ops.ADD, src=(UPat(Ops.LOAD), UPat(Ops.CONST)))

# Match ADD with a captured name (like regex groups)
UPat(Ops.ADD, src=(UPat.var("x"), UPat.var("y")))

# Match ADD of any type, any sources
UPat(Ops.ADD, name="add_op")
```

### Wildcards and Names

```python
# UPat.var("x") — match ANY UOp, capture it as "x"
UPat.var("x")                        # matches anything
UPat.var("x", dtype=dtypes.float)    # matches any float UOp

# UPat.cvar("c") — match a CONSTANT, capture as "c"
UPat.cvar("c")                       # matches CONST or VCONST

# UPat.const(dtype, value) — match specific constant value
UPat.const(None, 0)                  # matches 0 of any type
UPat.const(dtypes.float, 1.0)        # matches float 1.0
```

### Multiple Ops

```python
# Match either ADD or SUB
UPat({Ops.ADD, Ops.SUB}, name="op")

# Match any binary operation
UPat(GroupOp.Binary, name="binop")
```

### Commutative Matching

Lists vs tuples matter:

```python
# Tuple: order matters — first source must be LOAD, second must be CONST
UPat(Ops.ADD, src=(UPat(Ops.LOAD), UPat(Ops.CONST)))

# List: tries all permutations — either order works
UPat(Ops.ADD, src=[UPat(Ops.LOAD, name="a"), UPat(Ops.CONST, name="b")])
#  matches: ADD(LOAD, CONST) → a=LOAD, b=CONST
#  matches: ADD(CONST, LOAD) → a=LOAD, b=CONST  (swapped)
```

This is critical for commutative operations like ADD and MUL where `a+b == b+a`.

### The `.match()` Method

```python
pat = UPat(Ops.ADD, src=(UPat.var("x"), UPat.cvar("c")))

# Returns list of match dictionaries (empty list = no match)
results = pat.match(some_uop, {})
# results = [{"x": <UOp>, "c": <UOp>}] if matched
# results = [] if not matched
```

Matching is recursive — each sub-UPat must match the corresponding sub-UOp.

### Composition Helpers

UPat has builder methods that mirror UOp's API:

```python
# Match: CAST(ADD(x, y))
UPat(Ops.ADD, src=[UPat.var("x"), UPat.var("y")]).cast()

# Match: INDEX(BUFFER, RANGE)
UPat(Ops.BUFFER).index(UPat(Ops.RANGE))

# Match: LOAD(INDEX(BUFFER, ...))
UPat(Ops.BUFFER).index(UPat.var("idx")).load()
```

## PatternMatcher: Rule Sets

A `PatternMatcher` is a collection of `(UPat, function)` pairs. Given a UOp, it tries each pattern and calls the first matching function:

```python
simplify = PatternMatcher([
    # x + 0 → x
    (UPat(Ops.ADD, src=[UPat.var("x"), UPat.const(None, 0)]), lambda x: x),

    # x * 1 → x
    (UPat(Ops.MUL, src=[UPat.var("x"), UPat.const(None, 1)]), lambda x: x),

    # x * 0 → 0
    (UPat(Ops.MUL, src=[UPat.var("x"), UPat.cvar("c")]),
     lambda x, c: c if c.arg == 0 else None),

    # x - x → 0
    (UPat(Ops.SUB, src=(UPat.var("x"), UPat.var("x"))),
     lambda x: UOp.const(x.dtype, 0)),
])
```

### How Lookup Works

PatternMatcher builds an index by op type for fast dispatch:

```python
class PatternMatcher:
    def __init__(self, patterns):
        self.pdict: dict[Ops, list] = {}  # op → list of (pattern, match_fn, early_reject)

        for p, fxn in patterns:
            for op in p.op:
                self.pdict.setdefault(op, []).append([p, compiled_fn, p.early_reject])
```

When `rewrite(uop)` is called:
1. Look up `self.pdict[uop.op]` — O(1), only check patterns for this op type
2. **Early reject**: check if the UOp's source ops are a superset of the pattern's requirements. This eliminates most non-matches without full tree walking
3. Call the compiled match function
4. Return the first non-None result

```python
def rewrite(self, uop, ctx=None):
    if len(pats := self.pdict.get(uop.op, [])):
        ler = {u.op for u in uop.src}  # set of source op types
        for _, match, early_reject in pats:
            if not early_reject.issubset(ler):
                continue    # fast skip: required source types missing
            if (ret := match(uop, ctx)) is not None and ret is not uop:
                return ret  # matched and transformed!
    return None  # no pattern matched
```

### Context Parameter

Rules can take a `ctx` parameter for stateful transforms:

```python
# Pattern that uses external context
(UPat(Ops.BUFFER, name="b"), lambda ctx, b: ctx.get(b))

# Called as: graph_rewrite(sink, matcher, ctx=my_dict)
```

This is how the schedule cache normalization works — the `ctx` carries the buffer→parameter mapping.

### Composing PatternMatchers

PatternMatchers can be combined with `+`:

```python
pm_combined = pm_simplify + pm_lower + pm_optimize
# All rules from all three, tried in order
```

## graph_rewrite: The Engine

`graph_rewrite()` applies a PatternMatcher to an entire UOp graph:

```python
def graph_rewrite(sink: UOp, pm: PatternMatcher, ctx=None, bottom_up=False) -> UOp:
    ...
```

### Top-Down vs Bottom-Up

**Top-down** (default): starts at the sink (root) and rewrites downward.
- Good for: lowering high-level ops to low-level ones
- Each node is rewritten before its children

**Bottom-up** (`bottom_up=True`): starts at leaves and rewrites upward.
- Good for: constant folding, algebraic simplification
- Each node is rewritten after its children (children are already simplified)

```
Top-down:  process root first
              SINK         ← rewrite this first
             /    \
           ADD    STORE    ← then these
          /   \
        LOAD   CONST      ← then these

Bottom-up: process leaves first
              SINK         ← rewrite this last
             /    \
           ADD    STORE    ← then these
          /   \
        LOAD   CONST      ← rewrite these first
```

### Fixed-Point Iteration

`graph_rewrite` runs to a **fixed point** — it keeps applying rules until nothing changes. This means rules can create opportunities for other rules:

```python
# Round 1: fold constants
#   ADD(CONST(3), CONST(4)) → CONST(7)

# Round 2: simplify with new constant
#   MUL(x, CONST(7)) → ... (new optimization opportunities)

# Round 3: nothing changes → done
```

### How It Traverses

The rewrite engine does a topological traversal of the UOp graph (each UOp is visited once per pass). For each node:

1. Try to match against the PatternMatcher
2. If a match returns a new UOp, replace the node
3. After traversing all nodes, check if anything changed
4. If yes, traverse again. If no, done.

## Real Examples from TinyGrad

### Algebraic Simplification

From the symbolic math rewrite rules:

```python
# x + 0 → x
(UPat(Ops.ADD, src=[UPat.var("x"), UPat.const(None, 0)]), lambda x: x),

# x * 1 → x
(UPat(Ops.MUL, src=[UPat.var("x"), UPat.const(None, 1)]), lambda x: x),

# --x → x (double neg)
(UPat(Ops.NEG, src=(UPat(Ops.NEG, src=(UPat.var("x"),)),)), lambda x: x),

# bool < True → NOT bool
(UPat(Ops.CMPLT, dtypes.bool, (UPat.var("x"), UPat.const(dtypes.bool, True))),
 lambda x: x.ne(True)),
```

### Lowering High-Level to Low-Level

```python
# RELU: max(x, 0) stays as MAX in IR
# But SIGMOID: 1/(1+exp(-x)) gets decomposed:

(UPat(Ops.SIGMOID, src=(UPat.var("x"),)),
 lambda x: UOp.const(x.dtype, 1) / (UOp.const(x.dtype, 1) + (-x).exp())),
```

### Schedule Cache Normalization

From schedule.py (you saw this in [Pill 6](06-schedule-memory.md)):

```python
pm_pre_sched_cache = PatternMatcher([
    # Replace unique buffer IDs with sequential params
    (UPat(Ops.BUFFER, src=(UPat(Ops.UNIQUE), UPat(Ops.DEVICE)), name="b"),
     replace_input_buffer),

    # Strip BIND values (keep variable, drop constant)
    (UPat(Ops.BIND, src=(UPat(Ops.DEFINE_VAR), UPat(Ops.CONST)), name="b"),
     strip_bind),
])
```

### The Compilation Pipeline's 12+ Passes

From [Pill 2](02-compilation-pipeline.md), `full_rewrite_to_sink()` chains ~12 PatternMatchers:

```python
def full_rewrite_to_sink(k: UOp, ren: Renderer) -> UOp:
    k = graph_rewrite(k, pm_lowerer)           # 1. Lower high-level ops
    k = graph_rewrite(k, pm_late_rewrite)       # 2. Late rewrites
    k = graph_rewrite(k, pm_renderer)           # 3. Renderer-specific rules
    # ... more passes ...
```

Each pass is a self-contained PatternMatcher with its own rules. The output of one feeds into the next. This is the compilation pipeline as a **cascade of declarative rewrite rules**.

## Why Pattern Matching?

### vs. Visitor Pattern (traditional)

```python
# Traditional: manual tree walking
def optimize(node):
    if node.op == Ops.ADD:
        left = optimize(node.src[0])
        right = optimize(node.src[1])
        if right.op == Ops.CONST and right.arg == 0:
            return left
        if left.op == Ops.CONST and left.arg == 0:
            return right
        return UOp(Ops.ADD, node.dtype, (left, right))
    elif node.op == Ops.MUL:
        # ... 50 more lines of the same structure ...
```

```python
# TinyGrad: declare the rules
pm = PatternMatcher([
    (UPat(Ops.ADD, src=[UPat.var("x"), UPat.const(None, 0)]), lambda x: x),
    (UPat(Ops.ADD, src=[UPat.const(None, 0), UPat.var("x")]), lambda x: x),
    (UPat(Ops.MUL, src=[UPat.var("x"), UPat.const(None, 1)]), lambda x: x),
    # ... each rule is one line ...
])
result = graph_rewrite(root, pm)
```

Advantages of the pattern matching approach:
1. **Declarative**: rules say *what* to transform, not *how* to walk the tree
2. **Composable**: combine rule sets with `+`
3. **Self-documenting**: each rule is readable in isolation
4. **Safe**: graph_rewrite handles traversal, fixed-point, and graph rebuilding
5. **Debuggable**: with `TRACK_MATCH_STATS`, you see which rules fire and how often

### Tracing and Visualization

```bash
# Track which patterns fire
TRACK_MATCH_STATS=2 python3 my_script.py

# TinyGrad has a VIZ tool that shows rewrite traces graphically
VIZ=1 python3 my_script.py
```

Each match is recorded: which pattern matched, which UOp it transformed, and how long it took. This makes debugging the compiler surprisingly tractable.

## Performance: Early Reject

The early reject mechanism is what makes pattern matching fast despite hundreds of rules:

```python
# Pattern: ADD(LOAD, CONST)
# early_reject = {Ops.LOAD, Ops.CONST}

# When checking a UOp:
ler = {u.op for u in uop.src}  # e.g., {Ops.LOAD, Ops.MUL}

# early_reject.issubset(ler)?
# {Ops.LOAD, Ops.CONST} ⊆ {Ops.LOAD, Ops.MUL}?  → NO
# Skip this pattern without doing full matching
```

This converts O(patterns × tree_depth) matching into O(patterns) with very small constant factors. Most patterns are rejected by a single set operation.

## Compiled Patterns

For additional performance, UPat matching can be compiled from interpreted Python to optimized match functions:

```python
# Default: compiled matching (UPAT_COMPILE=1)
entry[1] = upat_compile(p, fxn)

# Fallback: interpreted matching
entry[1] = upat_interpret(p, fxn)
```

The compiled version generates a custom match function that avoids the general-purpose `UPat.match()` recursion. This matters when the same PatternMatcher is applied millions of times.

## Putting It Together: A Mini Compiler

Here's a complete mini-example showing how TinyGrad-style pattern matching works:

```python
from tinygrad.uop.ops import UOp, UPat, Ops, PatternMatcher, graph_rewrite
from tinygrad.dtype import dtypes

# Build a small computation: result = (x + 0) * 1
x = UOp(Ops.DEFINE_VAR, dtypes.float, arg=("x", 0, 100))
zero = UOp.const(dtypes.float, 0)
one = UOp.const(dtypes.float, 1)
result = (x + zero) * one

# Define simplification rules
pm = PatternMatcher([
    # x + 0 → x
    (UPat(Ops.ADD, src=[UPat.var("x"), UPat.const(None, 0)]),
     lambda x: x),
    # x * 1 → x
    (UPat(Ops.MUL, src=[UPat.var("x"), UPat.const(None, 1)]),
     lambda x: x),
])

# Apply rules
simplified = graph_rewrite(result, pm)
# simplified is just x — both identity operations removed!
```

This is exactly how TinyGrad's real compiler works, just at a larger scale (~500+ rules across all passes).

## Summary

- **UPat**: pattern templates for matching UOp trees. Supports wildcards, names, commutative matching, type constraints
- **PatternMatcher**: ordered list of `(UPat, function)` rules. Fast dispatch via op-type index + early reject
- **graph_rewrite**: applies a PatternMatcher to an entire graph, iterating to fixed point
- **Top-down**: for lowering (high → low level). **Bottom-up**: for simplification (constant folding, algebra)
- **List** src = permutation matching (commutative). **Tuple** src = ordered matching
- **Rules are composable**: `pm_a + pm_b` merges rule sets
- TinyGrad's entire compiler is chains of `graph_rewrite(graph, pm)` calls — ~12+ passes of declarative rules

---

**Previous**: [← Pill 7: BEAM Search](07-beam-search.md)
**Next**: [Pill 9: Jetson NV Backend, Part 1 →](09-jetson-nv-backend-pt1.md)
