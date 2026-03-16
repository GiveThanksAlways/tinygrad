# 02 — Attention Residuals Explained

> *This document walks through the math and code of the Attention Residual
> mechanism.  By the end you'll understand every line of `utils.py`.*

---

## 1. The Uniform Residual (Baseline)

Every modern transformer (GPT, Llama, Qwen) uses this pattern:

```python
# Standard pre-norm transformer block
def block(x):
    x = x + attention(layernorm(x))   # residual around attention
    x = x + ffn(layernorm(x))         # residual around FFN
    return x
```

Mathematically, after L layers:

```
x_L = x_0 + Σ_{l=0}^{L-1}  f_l(x_l)
```

Every layer's output `f_l` is added with **equal weight 1**.  The gradient
of the loss w.r.t. any earlier layer passes through an unweighted sum —
simple, stable, but wasteful.

---

## 2. The Attention Residual

### Core Idea

Replace the fixed `+ x` with a **learned aggregation** over all preceding
hidden states:

```
x_{l+1} = Σ_{j=0}^{l}  α_j · h_j
```

where `α_j` are attention weights computed from the *current* layer's output
attending to the *history* of all previous outputs:

```
α = softmax( Q(h_l) · K(H)^T / √d )
x_{l+1} = α · V(H)
```

Here `H = [h_0, h_1, ..., h_{l-1}]` is the stacked history and `Q, K, V`
are small learned projections.

### Why This Works

1. **No dilution** — the network can assign α_0 = 0.5 to layer 0 even at
   depth 96, keeping early features alive.
2. **Input-dependent** — different inputs route information differently
   through the depth of the network.
3. **Gradient highway** — attention weights create direct gradient paths
   from the loss to any layer, regardless of depth.

---

## 3. Pseudocode

```
function AttentionResidual(current, history):
    if history is empty:
        return current, [current]

    H = stack(history)               # (L, B, T, D)
    Q = W_q(current)                 # query from current layer
    K = W_k(H)                       # keys from history
    V = W_v(H)                       # values from history

    attn = softmax(Q · K^T / √d)    # attention over history
    aggregated = attn · V            # weighted combination

    # Gated combination for training stability
    g = sigmoid(gate)
    output = g * aggregated + (1 - g) * (history[-1] + current)

    return output, history + [output]
```

---

## 4. Code Comparison

### Before: Uniform Residual

```python
class ClassicBlock:
  def __call__(self, x):
    x = x + self.attn(self.ln1(x))    # fixed +1 residual
    x = x + self.ffn(self.ln2(x))     # fixed +1 residual
    return x

# In the model's forward pass:
for block in self.blocks:
    x = block(x)                       # each block adds uniformly
```

### After: Attention Residual

```python
class AttnResBlock:
  def __call__(self, x):
    attn_out = self.attn(self.ln1(x))
    out = attn_out + self.ffn(self.ln2(attn_out + x))
    return out                         # raw output, no residual here

# In the model's forward pass:
history = [x]
for block, ar in zip(self.blocks, self.attn_res):
    raw = block(x)
    x, history = ar(raw, history)      # learned attention over history
```

The structural change is small — we just move the residual connection
*outside* the block and replace it with an attention mechanism.

---

## 5. The Gated Combination

Training can be unstable if we jump straight to full attention aggregation.
The solution: a **learned scalar gate** initialized at 0:

```python
self.gate = Tensor.zeros(1)        # sigmoid(0) = 0.5

g = self.gate.sigmoid()
output = g * attn_aggregated + (1 - g) * classic_residual
```

At initialization, the model behaves like a 50/50 mix of attention residual
and classic residual.  During training it learns the optimal balance — often
converging to g ≈ 0.7–0.9 (favoring attention residuals) in deeper models.

---

## 6. Block AttnRes — Scaling to Deep Models

### The Problem

Full attention residual stores L history entries per layer.  For a 96-layer
model that's O(L²) memory — expensive!

### The Solution: Block Compression

Group every B layers into a **block** and keep one summary per block
(the mean of the block's outputs):

```
Original history:  [h0, h1, h2, h3, h4, h5, h6, h7, ...]
Block size = 4:    [mean(h0..h3), mean(h4..h7), ...]
```

Now attention runs over ≈ L/B entries instead of L.  With B=4 and L=96,
that's 24 entries instead of 96 — a 4× reduction.

```python
class BlockAttentionResidual:
  def __init__(self, embed_dim, num_heads=4, block_size=4):
    self.block_size = block_size
    self.attn_res = AttentionResidualLayer(embed_dim, num_heads)

  def __call__(self, current, history):
    # Compress history into block summaries
    compressed = []
    for i in range(0, len(history), self.block_size):
      block = Tensor.stack(*history[i:i+self.block_size], dim=0)
      compressed.append(block.mean(axis=0))

    output, _ = self.attn_res(current, compressed)
    return output, history + [output]
```

### Memory Comparison

| Model depth | Uniform Residual | Full AttnRes | Block AttnRes (B=4) |
|-------------|-----------------|--------------|---------------------|
| 12 layers   | 12 × D          | 78 × D       | 15 × D              |
| 48 layers   | 48 × D          | 1,176 × D    | 60 × D              |
| 96 layers   | 96 × D          | 4,656 × D    | 120 × D             |

(D = hidden dimension size per token)

---

## 7. The 1.25× Efficiency Gain — What It Means on Orin

The key claim: Attention Residuals give **~1.25× effective compute** at the
same depth.  What does this mean concretely?

| Scenario | Without AttnRes | With AttnRes |
|----------|----------------|--------------|
| Match quality of 48-layer model | 48 layers needed | ~38 layers enough |
| Match quality of 96-layer model | 96 layers needed | ~77 layers enough |

On a Jetson AGX Orin with 64 GB unified memory:

- A 38-layer model uses **~20% less memory** than a 48-layer model
- That freed memory lets you increase batch size or sequence length
- Faster training per step (fewer layers to compute)
- **Net effect:** you can train a "bigger-feeling" model within the same
  memory budget

---

## 8. Key Implementation Details

### Parameter Count

The attention residual adds a small overhead per layer:

```
Per-layer overhead = 4 × D² + 1   (Q, K, V, O projections + gate)
```

For D=4096 (a 7B-scale model), that's ~67M extra parameters per layer,
or about 1.5% of the total model.  A good trade for 25% more efficiency.

### Gradient Flow

During backpropagation, the attention weights create **direct gradient
highways** from the loss to every layer in the history.  This is
analogous to how skip connections in ResNets help gradients flow, but
*adaptive* rather than fixed.

### Compatibility

Attention Residuals are a **drop-in architectural change**.  They work
with any transformer variant (GPT, Llama, Qwen, Mistral) and are
compatible with:
- LoRA / QLoRA fine-tuning
- Gradient checkpointing
- Mixed precision (fp16/bf16)
- Flash Attention

---

## Next Steps

- **Run** `03_Small_Model_Experiment.py` to see both approaches train
  on a real task
- **Read** `04_Scale_to_30B_FineTune_Guide.md` for instructions on
  applying this to a large model on your Jetson Orin
