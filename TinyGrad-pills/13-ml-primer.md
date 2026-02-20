# Pill 13: Machine Learning Primer (in TinyGrad)

## Why This Pill

You know ioctls and kernel drivers. You've followed the pills on UOps, scheduling, and GPU backends. But you might be fuzzy on **what these frameworks are actually computing** and why the math is shaped the way it is.

This pill reviews core ML concepts through TinyGrad's API — enough to read model code and understand what's happening at each layer.

## The Training Loop

Every neural network training follows the same 5-step loop:

```python
from tinygrad import Tensor
from tinygrad.nn import optim, state

Tensor.training = True
model = MyModel()
optimizer = optim.Adam(state.get_parameters(model), lr=0.001)

for batch in dataloader:
    # 1. Forward pass — compute predictions
    pred = model(batch.input)
    
    # 2. Loss — how wrong are we?
    loss = (pred - batch.target).square().mean()
    
    # 3. Backward pass — compute gradients
    loss.backward()
    
    # 4. Update — adjust weights
    optimizer.step()
    
    # 5. Zero gradients — reset for next batch
    optimizer.zero_grad()
```

That's it. Everything else — transformers, diffusion models, GANs — is variations on these 5 steps.

### What Each Step Does

| Step | What | TinyGrad Mechanism |
|------|------|--------------------|
| Forward | Run data through layers | Tensor ops build UOp graph (lazy) |
| Loss | Scalar measuring error | Standard ops (MSE, cross-entropy) |
| Backward | Compute ∂loss/∂weight for every weight | `loss.backward()` → reverse-mode autograd |
| Update | weight -= lr × gradient | Optimizer modifies parameter tensors |
| Zero grad | Clear old gradients | `optimizer.zero_grad()` sets `.grad = None` |

**Key insight**: Steps 1-3 are lazy in TinyGrad. Nothing runs on the GPU until `.realize()` or `.backward()` is called (backward calls realize internally).

## Tensors and Shapes

Every piece of data in ML is a tensor — a multi-dimensional array.

```python
from tinygrad import Tensor

scalar = Tensor(3.14)                    # shape: ()
vector = Tensor([1, 2, 3])              # shape: (3,)
matrix = Tensor([[1, 2], [3, 4]])       # shape: (2, 2)
batch  = Tensor.randn(32, 3, 224, 224)  # shape: (32, 3, 224, 224)
#                       ↑  ↑   ↑    ↑
#                   batch channels height width
```

Convention for dimensions:
- **B** (batch): how many samples processed in parallel
- **C** (channels): features per spatial position (e.g., RGB=3)
- **H, W** (height, width): spatial dimensions
- **T** (time/tokens): sequence length
- **D** (dim): embedding/hidden size

### Broadcasting

TinyGrad (like NumPy) automatically expands dimensions for elementwise ops:

```python
A = Tensor.randn(32, 768)    # (32, 768)
b = Tensor.randn(768)        # (768,)
result = A + b               # (32, 768) — b broadcast across batch
```

Rules: dimensions are aligned from the right. Size-1 (or missing) dimensions get expanded.

## Layers (nn Module)

TinyGrad's `nn` module provides standard layers. No `Module` base class needed — layers are just plain Python classes with `__call__` and tensor attributes.

### Linear (Fully Connected)

The most fundamental layer: $y = xW^T + b$

```python
from tinygrad import nn

layer = nn.Linear(768, 256)    # 768→256
# layer.weight: Tensor (256, 768)
# layer.bias:   Tensor (256,)

x = Tensor.randn(32, 768)     # batch of 32, 768 features each
y = layer(x)                   # shape: (32, 256)
```

Under the hood: `x.linear(self.weight.transpose(), self.bias)` — a matmul + bias add.

**During LLM decode**: batch=1, so this becomes a **matvec** ([Pill 11](11-matvec-heuristic.md)): one vector × one matrix.

### Conv2d (Convolution)

Slides a learned filter over spatial dimensions:

```python
conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
# conv.weight: Tensor (64, 3, 3, 3)
# conv.bias:   Tensor (64,)

img = Tensor.randn(1, 3, 224, 224)   # 1 RGB image
out = conv(img)                        # shape: (1, 64, 224, 224)
```

Each output channel is a dot product between the filter and a local region of the input.

### LayerNorm

Normalizes each sample to zero-mean, unit-variance, then rescales:

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

```python
norm = nn.LayerNorm(768)
x = Tensor.randn(32, 10, 768)   # batch=32, seq=10, dim=768
y = norm(x)                      # same shape, normalized over last dim
```

Critical for transformers — stabilizes training by keeping activations in a good numerical range.

### Embedding

Lookup table: integer → vector.

```python
embed = nn.Embedding(50257, 768)   # vocabulary of 50K tokens, 768-dim
# embed.weight: Tensor (50257, 768)

token_ids = Tensor([1, 42, 1337, 7])  # 4 tokens
vectors = embed(token_ids)            # shape: (4, 768)
```

No math — just indexing. `embed.weight[token_ids]`.

## Loss Functions

The loss measures how wrong the model is. It must be a **scalar** (single number) so we can take its gradient.

### Mean Squared Error (MSE)

For regression (predicting continuous values):

$$L = \frac{1}{N} \sum_i (y_i - \hat{y}_i)^2$$

```python
pred = model(x)           # (32, 1)
target = Tensor(labels)   # (32, 1)
loss = (pred - target).square().mean()
```

### Cross-Entropy

For classification (predicting categories):

$$L = -\frac{1}{N} \sum_i \log\left(\frac{e^{z_{i,c}}}{\sum_j e^{z_{i,j}}}\right)$$

Where $c$ is the correct class. In TinyGrad:

```python
logits = model(x)                           # (32, 10) — 10 classes
loss = logits.sparse_categorical_crossentropy(labels)
```

Cross-entropy combines softmax (turn logits into probabilities) and negative log-likelihood (penalize low probability on correct class).

## Autograd: How Backward Works

TinyGrad implements **reverse-mode automatic differentiation** (backpropagation).

### The Chain Rule

For $f(g(x))$: $\frac{\partial f}{\partial x} = \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial x}$

Applied recursively through the computation graph:

```
Forward:  x → [Linear] → h → [ReLU] → a → [Linear] → y → [Loss] → L
                                                                     ↓
Backward: dx ← [∂L/∂h] ← dh ← [∂L/∂a] ← da ← [∂L/∂y] ← dy ← [dL=1]
```

### TinyGrad's Implementation

```python
# 1. Forward: builds UOp computation graph (lazy)
y = (x @ W).relu()
loss = y.sum()

# 2. backward() does:
#    a. Toposort all UOps reachable from loss
#    b. Filter to tensors with requires_grad=True
#    c. Call compute_gradient() — reverse traversal
#    d. Accumulate gradients into .grad attribute
loss.backward()

# 3. Gradients are now populated
print(W.grad)  # ∂loss/∂W — same shape as W
```

The `compute_gradient` function walks the UOp graph in reverse topological order, applying the chain rule at each op. Each UOp type knows its own derivative (e.g., `MUL`'s gradient w.r.t. first arg is the second arg).

### New API: `gradient()`

TinyGrad also has a functional gradient API:

```python
x = Tensor.eye(3)
y = Tensor([[2.0, 0, -2.0]])
z = y.matmul(x).sum()

dx, dy = z.gradient(x, y)  # explicit targets
```

This is more flexible — you choose which gradients to compute.

## Optimizers

Optimizers use gradients to update weights. TinyGrad implements them as pure functions on tensors (with optimizer fusion for performance).

### SGD (Stochastic Gradient Descent)

The simplest: $w \leftarrow w - \eta \cdot \nabla L$

```python
opt = optim.SGD(state.get_parameters(model), lr=0.01, momentum=0.9)
```

With momentum: remembers previous gradient direction, smoothing updates.

### Adam

Adaptive learning rate per-parameter using first and second moment estimates:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$w \leftarrow w - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

```python
opt = optim.Adam(state.get_parameters(model), lr=0.001)
```

Adam adapts step size per parameter — parameters with large gradients get smaller steps, sparse parameters get larger steps. De facto standard for transformers.

### AdamW

Adam with **decoupled weight decay** — subtracts a fraction of the weight directly:

```python
opt = optim.AdamW(state.get_parameters(model), lr=0.001, weight_decay=0.01)
```

### Optimizer Fusion

TinyGrad's `FUSE_OPTIM` (default on) concatenates all parameters into one big tensor, runs the optimizer step as a single kernel, then splits back. Fewer kernel launches = faster.

```python
# Without fusion: N kernels (one per parameter)
# With fusion:    1 kernel (all parameters at once)
```

## Model State: Save & Load

```python
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict

# Save
state_dict = get_state_dict(model)
safe_save(state_dict, "model.safetensors")

# Load
state_dict = safe_load("model.safetensors")
load_state_dict(model, state_dict)
```

TinyGrad uses the **safetensors** format (same as Hugging Face). `get_state_dict` recursively walks Python objects collecting all `Tensor` attributes into a flat `{name: Tensor}` dict.

### No Module Base Class

Unlike PyTorch's `nn.Module`, TinyGrad layers are just classes with tensor attributes:

```python
class MyModel:
    def __init__(self):
        self.l1 = nn.Linear(784, 128)
        self.l2 = nn.Linear(128, 10)
    
    def __call__(self, x):
        return self.l2(self.l1(x).relu())

# get_state_dict walks __dict__ recursively
# Returns: {"l1.weight": ..., "l1.bias": ..., "l2.weight": ..., "l2.bias": ...}
```

No `register_parameter`, no `register_buffer`, no `forward()` method. Just Python.

## The Transformer (What LLMs Are)

Since our benchmarks are all LLMs, here's the 30-second version:

### Architecture

```
Input tokens → [Embedding] → [Transformer Block × N] → [Output Head] → Next token
                                      ↓
                    [LayerNorm → Attention → LayerNorm → FFN]
```

### Self-Attention

The key innovation: every token looks at every other token to decide what's important.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Where:
- **Q** (query): "what am I looking for?"
- **K** (key): "what do I contain?"
- **V** (value): "what information to pass along?"

All three are linear projections of the same input (for self-attention).

### Feed-Forward Network (FFN)

Two linear layers with an activation:

$$\text{FFN}(x) = W_2 \cdot \text{activation}(W_1 x + b_1) + b_2$$

Modern LLMs use SwiGLU: $\text{SwiGLU}(x) = (xW_1) \otimes \text{silu}(xW_{\text{gate}})$

### Why Decode is Memory-Bound

During generation, the model produces one token at a time. For each token:
1. Run the full model forward (all layers)  
2. Every linear layer is batch_size=1 → **matvec** (not matmul)
3. Must read entire weight matrix from memory for one vector output
4. ~1 GB of weights read, ~1 KB of output → arithmetic intensity ≈ 2

This is why memory bandwidth, not compute, determines tok/s ([Pill 12](12-benchmarking.md)).

## Inference vs Training

| | Training | Inference |
|---|---------|-----------|
| **Goal** | Learn weights | Generate output |
| **Gradients** | Yes (backward pass) | No |
| **Batch size** | Large (32-2048) | Small (1-8) |
| **Bottleneck** | Compute (matmul) | Memory (matvec) |
| **Key metric** | Samples/sec, loss | Tokens/sec, latency |
| **TinyGrad flag** | `Tensor.training = True` | `Tensor.training = False` |

Our Jetson work is all inference — running pretrained models to generate text.

## Quantization (Why Q6_K)

Full-precision weights (FP16) use 2 bytes per weight. For a 1B parameter model, that's 2 GB. **Quantization** reduces this:

| Format | Bits/weight | 1B Model Size | Quality Impact |
|--------|------------|---------------|----------------|
| FP16 | 16 | 2.0 GB | Baseline |
| Q8_0 | 8 | 1.0 GB | Near-lossless |
| Q6_K | 6.5 | 0.81 GB | Minimal loss |
| Q4_K_M | 4.5 | 0.56 GB | Slight degradation |
| Q2_K | 2.6 | 0.33 GB | Noticeable loss |

Q6_K groups weights into blocks, stores a per-block scale factor, and quantizes individual values to 6 bits. The "K" means k-quant — uses different quantization for different parts of the model (attention vs FFN) based on importance.

**On Orin**: smaller model = less memory to read per token = more tok/s. Q6_K is the sweet spot between quality and throughput.

## TinyGrad vs PyTorch: Key Differences

| Feature | PyTorch | TinyGrad |
|---------|---------|----------|
| Execution | Eager (immediate) | Lazy (deferred) |
| Module system | `nn.Module` base class | Plain Python classes |
| Autograd | Tape-based | UOp graph traversal |
| Device mgmt | `.to(device)` | Device string in constructor |
| Compilation | `torch.compile` (optional) | Always compiled |
| Model format | `.pt` / safetensors | safetensors |
| Optimizer | Class per optimizer | LARS/LAMB unification |
| Size | ~3M lines | ~10K lines |

The biggest practical difference: **TinyGrad compiles everything**. There's no "eager mode." Every operation builds a lazy graph that gets compiled and optimized before execution.

## Putting It Together

Here's a minimal training example that ties everything together:

```python
from tinygrad import Tensor
from tinygrad.nn import optim, state
import tinygrad.nn as nn

# Model
class MLP:
    def __init__(self):
        self.l1 = nn.Linear(784, 128)
        self.l2 = nn.Linear(128, 10)
    def __call__(self, x):
        return self.l2(self.l1(x).relu())

model = MLP()
opt = optim.Adam(state.get_parameters(model), lr=1e-3)

# Training
Tensor.training = True
for step in range(1000):
    x = Tensor.randn(64, 784)       # fake data
    y = Tensor.randint(64, high=10)  # fake labels
    
    loss = model(x).sparse_categorical_crossentropy(y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    
    if step % 100 == 0:
        print(f"step {step}, loss {loss.item():.4f}")

# Save
state.safe_save(state.get_state_dict(model), "mlp.safetensors")
```

What happens under the hood for each iteration:
1. `model(x)` → builds UOp graph (two matmuls, relu, cross-entropy)
2. `loss.backward()` → extends graph with gradient computation, then `realize()` → schedule → compile → run on GPU
3. `opt.step()` → fused optimizer kernel updates all weights in one launch
4. `opt.zero_grad()` → sets `.grad = None` (Python-side, no GPU work)

## Summary

- **Training loop**: forward → loss → backward → update → zero_grad
- **Layers**: Plain Python classes with Tensor attributes — no `nn.Module` needed
- **Autograd**: Reverse-mode differentiation via UOp graph traversal
- **Optimizers**: SGD, Adam, AdamW, LARS, LAMB, Muon — unified implementation
- **State**: `get_state_dict` / `load_state_dict` with safetensors format
- **Transformers**: Embedding → N × (Attention + FFN) → output head
- **LLM decode**: Memory-bound because batch=1 → every layer is matvec
- **Quantization**: Q6_K = 6.5 bits/weight, sweet spot for Orin
- **Key difference from PyTorch**: Everything is lazy and compiled, always

---

**Previous**: [← Pill 12: Benchmarking & Performance](12-benchmarking.md)
**Next**: [Pill 14: Contributing to TinyGrad →](14-contributing.md)
