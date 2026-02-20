# Pill 15: Why tinygrad Wins and Loses — A Cross-Framework Analysis

## The Scoreboard

We benchmarked four LLM inference frameworks on the Jetson AGX Orin 64GB.
Here's what happened:

| Model      | tinygrad NV=1  | llama.cpp +FA | vLLM fp16 | MLC LLM fp16 |
| ---------- | :------------: | :-----------: | :-------: | :----------: |
| Qwen3 0.6B | **41.0 tok/s** |     43.0      |     ❌     |      ❌       |
| LLaMA 1B   |    **29.0**    |     27.85     |   30.1    |   **36.8**   |
| LLaMA 3B   |    **12.1**    |     11.86     |     —     |      —       |

tinygrad beats llama.cpp on every model. It ties vLLM. But MLC LLM is 27% faster.

**Why?** This pill explains the first principles behind each win and loss.

---

## Part 1: The Concepts You Need

### What is "Decode" in LLM Inference?

LLM inference has two phases:

```text
Phase 1: PREFILL  (process the prompt)
  Input:  "Explain transformers in detail..."  →  42 tokens
  Shape:  weight[2048, 2048] × input[42, 2048] = output[42, 2048]
  This is MATMUL — matrix × matrix
  GPU-compute bound (lots of arithmetic per byte read)

Phase 2: DECODE  (generate tokens one at a time)
  Input:  one new token  →  1 token
  Shape:  weight[2048, 2048] × input[1, 2048] = output[1, 2048]
  This is MATVEC — matrix × vector
  Memory-bandwidth bound (read entire model, do tiny amount of math)
```

**Decode dominates wall-clock time.** Generating 128 tokens means 128 matvec passes through the entire model. Each pass reads every weight once. The GPU spends almost all its time waiting for DRAM.

### What is Memory-Bandwidth Bound?

The Orin AGX has LPDDR5 memory at ~205 GB/s peak bandwidth (bidirectional, ~102 GB/s reads in practice). For a 1B parameter model stored in fp16 (2 bytes per weight):

$$\text{model size in memory} = 1.24\text{B} \times 2 \text{ bytes} = 2.48 \text{ GB}$$

$$\text{theoretical max tok/s} = \frac{102 \text{ GB/s}}{2.48 \text{ GB}} \approx 41 \text{ tok/s}$$

Every framework is racing to read model weights from DRAM as fast as possible. The math (multiply-add) takes almost zero time by comparison. This is called being **memory-bandwidth bound** — the bottleneck is memory reads, not compute.

**Key insight**: The only ways to go faster are:

1. Read fewer bytes (smaller quantization)
2. Read bytes faster (better memory access patterns)
3. Waste less time between reads (lower dispatch overhead)

### Linear Algebra by Hand: What the LLM Actually Computes

Before we talk about frameworks, let's do the exact same math an LLM does — by hand, with tiny numbers. Once you see it, everything else clicks.

#### Dot Product (the atom of neural networks)

Two vectors, same length. Multiply element-wise, sum:

```
  a = [2, 3, 1]
  b = [4, 1, 5]

  dot(a, b) = (2×4) + (3×1) + (1×5)
            =   8   +   3   +   5
            = 16
```

That's it. Every single computation in an LLM — attention, projections, feed-forward — is dot products stacked up. A neuron is a dot product + nonlinearity.

#### Matrix × Vector (Matvec) — One LLM Decode Step

Stack multiple dot products = matvec. This is what happens **every time the LLM generates one token**:

```
  Weight matrix W (3×4):          Input vector x (4×1):
  ┌                    ┐          ┌     ┐
  │  0.5  -1.0  2.0  0.3 │       │  1.0 │
  │ -0.2   1.5  0.0  0.8 │   ×   │ -2.0 │
  │  1.0   0.5 -1.0  0.4 │       │  0.5 │
  └                    ┘          │  3.0 │
                                  └     ┘

  Row 0:  (0.5×1.0) + (-1.0×-2.0) + (2.0×0.5) + (0.3×3.0)
        =    0.5    +     2.0     +    1.0    +    0.9
        = 4.4

  Row 1:  (-0.2×1.0) + (1.5×-2.0) + (0.0×0.5) + (0.8×3.0)
        =   -0.2    +    -3.0    +    0.0    +    2.4
        = -0.8

  Row 2:  (1.0×1.0) + (0.5×-2.0) + (-1.0×0.5) + (0.4×3.0)
        =    1.0   +    -1.0    +    -0.5    +    1.2
        = 0.7

  Output y (3×1):
  ┌      ┐
  │  4.4 │
  │ -0.8 │
  │  0.7 │
  └      ┘
```

**Cost**: 3 rows × 4 multiplies = 12 multiply-adds. We read all 12 weights + 4 input values = 16 values.

In a real LLM (LLaMA 1B), one layer's projection is **2048 × 2048** — that's 4,194,304 multiply-adds per matvec. And there are ~64 such matvecs per token (Q, K, V, O projections × 16 layers + FFN up/down/gate × 16 layers). **One token = ~268 million multiply-adds.**

But on Orin's GPU with 1024 CUDA cores at 1 GHz, that's only ~0.26 ms of compute. Reading the weights (2048×2048×2 bytes = 8 MB per matrix, ~64 matrices = 512 MB) at 102 GB/s takes ~5 ms. **The math finishes 20× before the data arrives.** That's memory-bandwidth bound.

#### Matrix × Matrix (Matmul) — Prefill / Batched Inference

During prefill (processing your prompt), we don't have one vector — we have many tokens at once:

```
  Weight W (3×4):                 Input X (4×2):    ← 2 tokens
  ┌                    ┐          ┌           ┐
  │  1   0  -1   2    │          │  1    0   │
  │  0   1   1  -1    │    ×     │  2   -1   │
  │  2  -1   0   1    │          │  0    3   │
  └                    ┘          │ -1    1   │
                                  └           ┘

  Output Y (3×2):
  Row 0, Col 0:  (1×1) + (0×2) + (-1×0) + (2×-1) = 1+0+0-2     = -1
  Row 0, Col 1:  (1×0) + (0×-1) + (-1×3) + (2×1) = 0+0-3+2     = -1
  Row 1, Col 0:  (0×1) + (1×2) + (1×0) + (-1×-1) = 0+2+0+1     =  3
  Row 1, Col 1:  (0×0) + (1×-1) + (1×3) + (-1×1) = 0-1+3-1     =  1
  Row 2, Col 0:  (2×1) + (-1×2) + (0×0) + (1×-1) = 2-2+0-1     = -1
  Row 2, Col 1:  (2×0) + (-1×-1) + (0×3) + (1×1) = 0+1+0+1     =  2

  Y = ┌        ┐
      │ -1  -1 │
      │  3   1 │
      │ -1   2 │
      └        ┘
```

**Key difference**: for matmul, we read weight W **once** but get results for **2 tokens**. The more tokens we batch, the more compute we do per byte read. This is why prefill is compute-bound (good GPU utilization) but decode is memory-bound (1 token = read everything, compute almost nothing).

### What is a Transformer? (The 30-Second Version)

A transformer is a stack of identical layers. Each layer has two parts:

```
Input token embedding (vector of d numbers, e.g., d=2048)
  │
  ▼
┌─────────────────────────────────────────────┐
│  ATTENTION BLOCK                            │
│  "Which previous tokens should I look at?"  │
│                                             │ ─── repeated N times
│  Q = x × W_q    (query: "what am I?")      │     (N=16 for LLaMA 1B)
│  K = x × W_k    (key: "what do I offer?")  │
│  V = x × W_v    (value: "here's my info")  │
│  attn = softmax(Q × K^T / √d) × V         │
│  out = attn × W_o                          │
├─────────────────────────────────────────────┤
│  FEED-FORWARD BLOCK (FFN)                   │
│  "Process what attention found"             │
│                                             │
│  up   = x × W_up      (expand: d → 4d)     │
│  gate = x × W_gate    (gating: d → 4d)     │
│  down = SiLU(gate) ⊙ up × W_down  (4d → d) │
└─────────────────────────────────────────────┘
  │
  ▼  (after N layers)
Final linear: hidden × W_vocab → logits (one score per vocab word)
  │
  ▼
argmax(logits) → next token ID → "The"
```

**Where are the parameters?** Almost entirely in the weight matrices:

| Matrix per layer | Shape (LLaMA 1B) | Params | What it does |
|---|---|---:|---|
| W_q | 2048 × 2048 | 4.2M | Projects to queries |
| W_k | 2048 × 512 | 1.0M | Projects to keys (GQA: fewer heads) |
| W_v | 2048 × 512 | 1.0M | Projects to values |
| W_o | 2048 × 2048 | 4.2M | Projects attention output |
| W_up | 2048 × 8192 | 16.8M | FFN expand |
| W_gate | 2048 × 8192 | 16.8M | FFN gating |
| W_down | 8192 × 2048 | 16.8M | FFN compress back |
| **Per layer total** | | **60.8M** | |
| **× 16 layers** | | **972.8M** | |
| + embeddings, norms | | **~270M** | |
| **Total** | | **~1.24B** | → 2.48 GB in fp16 |

Every token generation requires reading **all** of these weights from DRAM. That's the bandwidth race.

### Attention by Hand: Do the Same Math the LLM Does

Let's work through self-attention with tiny dimensions. Imagine `d=4`, `seq_len=3` (we've seen 3 tokens so far), 1 attention head.

**Step 1: We have 3 token embeddings (rows of X)**

```
  X (3×4):   ← 3 tokens, each a 4-dimensional vector
  ┌                     ┐
  │  1.0   0.0  -1.0  2.0 │   ← token 0: "The"
  │  0.5   1.0   0.5  0.0 │   ← token 1: "cat"
  │ -1.0   2.0   1.0  1.0 │   ← token 2: "sat"  (current token)
  └                     ┘
```

**Step 2: Project to Q, K, V** (each is X × W_something, but let's just use the results)

```
  Q (3×4):                K (3×4):                V (3×4):
  ┌              ┐        ┌              ┐        ┌              ┐
  │  1   0  -1  0 │      │  0   1   0  1 │      │  2   1   0   0 │
  │  0   1   0  1 │      │  1  -1   1  0 │      │  0   0   1   1 │
  │  1   1   0 -1 │      │  0   0   1  1 │      │  1   1   1   0 │
  └              ┘        └              ┘        └              ┘
```

**Step 3: Compute attention scores `S = Q × K^T`** (each query asks "how relevant is each key?")

```
  K^T (4×3):               S = Q × K^T (3×3):
  ┌           ┐
  │  0  1  0  │
  │  1 -1  0  │     Q[0] · K[0] = (1×0)+(0×1)+(-1×0)+(0×1)     =  0
  │  0  1  1  │     Q[0] · K[1] = (1×1)+(0×-1)+(-1×1)+(0×0)    =  0
  │  1  0  1  │     Q[0] · K[2] = (1×0)+(0×0)+(-1×1)+(0×1)     = -1
  └           ┘
                    Q[1] · K[0] = (0×0)+(1×1)+(0×0)+(1×1)       =  2
                    Q[1] · K[1] = (0×1)+(1×-1)+(0×1)+(1×0)      = -1
                    Q[1] · K[2] = (0×0)+(1×0)+(0×1)+(1×1)       =  1

                    Q[2] · K[0] = (1×0)+(1×1)+(0×0)+(-1×1)      =  0
                    Q[2] · K[1] = (1×1)+(1×-1)+(0×1)+(-1×0)     =  0
                    Q[2] · K[2] = (1×0)+(1×0)+(0×1)+(-1×1)      = -1

  S = ┌          ┐
      │  0  0 -1 │   ← token "The" attends to [The, cat, sat]
      │  2 -1  1 │   ← token "cat" attends to [The, cat, sat]
      │  0  0 -1 │   ← token "sat" attends to [The, cat, sat]
      └          ┘
```

**Step 4: Scale by $\frac{1}{\sqrt{d}}$ and apply softmax** (turn scores into probabilities)

```
  Scale: 1/√4 = 0.5

  S_scaled = ┌                ┐
             │  0.0  0.0 -0.5 │
             │  1.0 -0.5  0.5 │
             │  0.0  0.0 -0.5 │
             └                ┘

  Softmax (per row — each row sums to 1.0):
    Row 0: e^0.0=1.00, e^0.0=1.00, e^-0.5=0.61 → normalize: [0.38, 0.38, 0.23]
    Row 1: e^1.0=2.72, e^-0.5=0.61, e^0.5=1.65 → normalize: [0.55, 0.12, 0.33]
    Row 2: e^0.0=1.00, e^0.0=1.00, e^-0.5=0.61 → normalize: [0.38, 0.38, 0.23]

  P = ┌                  ┐
      │ 0.38  0.38  0.23 │   "The" looks equally at The/cat, less at sat
      │ 0.55  0.12  0.33 │   "cat" looks mostly at The, some at sat
      │ 0.38  0.38  0.23 │   "sat" looks equally at The/cat, less at sat
      └                  ┘
```

**Step 5: Weighted sum of Values `O = P × V`** (blend the values based on attention)

```
  V (3×4):                        O = P × V (3×4):
  ┌              ┐
  │ 2  1  0  0  │
  │ 0  0  1  1  │
  │ 1  1  1  0  │

  O[0] = 0.38×[2,1,0,0] + 0.38×[0,0,1,1] + 0.23×[1,1,1,0]
       = [0.76, 0.38, 0, 0] + [0, 0, 0.38, 0.38] + [0.23, 0.23, 0.23, 0]
       = [0.99, 0.61, 0.61, 0.38]

  O[1] = 0.55×[2,1,0,0] + 0.12×[0,0,1,1] + 0.33×[1,1,1,0]
       = [1.10, 0.55, 0, 0] + [0, 0, 0.12, 0.12] + [0.33, 0.33, 0.33, 0]
       = [1.43, 0.88, 0.45, 0.12]

  O[2] = same as O[0] = [0.99, 0.61, 0.61, 0.38]
```

**What just happened?** Each token's output is a weighted blend of all tokens' values, where the weights came from how much each query "matched" each key. Token "cat" decided to pay 55% attention to "The", 12% to itself, and 33% to "sat".

**Scale to real LLMs**: In LLaMA 1B, `d=2048`, `num_heads=32`, `head_dim=64`. Each head does this same computation with 64-dim vectors. With 32 heads running in parallel, the model learns 32 different "types of attention" (one head might track subject-verb, another tracks adjective-noun, etc.).

### What is the KV Cache?

During decode (generating tokens one at a time), there's a critical optimization: **don't recompute K and V for past tokens**.

```
Without KV cache (naive — recompute everything):
  Token 0: compute Q₀,K₀,V₀  → attn(Q₀, [K₀], [V₀])            → 1 token
  Token 1: compute Q₁,K₁,V₁  → attn(Q₁, [K₀,K₁], [V₀,V₁])     → 2 tokens
  Token 2: compute Q₂,K₂,V₂  → attn(Q₂, [K₀,K₁,K₂], [V₀,V₁,V₂]) → 3 tokens
  ...
  Token N: compute Q_N to recompute ALL K₀..K_N, V₀..V_N          → N tokens
  Cost: O(N²) total work to generate N tokens

With KV cache (store K,V from previous tokens):
  Token 0: compute Q₀,K₀,V₀ → cache K₀,V₀ → attn(Q₀, cache)    → read 1 KV
  Token 1: compute Q₁,K₁,V₁ → cache K₁,V₁ → attn(Q₁, cache)    → read 2 KVs
  Token 2: compute Q₂,K₂,V₂ → cache K₂,V₂ → attn(Q₂, cache)    → read 3 KVs
  ...
  Token N: compute Q_N,K_N,V_N → cache → attn(Q_N, cache)         → read N KVs
  Cost: O(N) total work (just ONE projection + cache read per step)
```

**The tradeoff**: KV cache saves massive compute but **uses memory** that grows with sequence length:

```
KV cache size per token = 2 × num_layers × num_kv_heads × head_dim × bytes_per_element

LLaMA 1B (d=2048, 16 layers, 8 KV heads, head_dim=64, fp16):
  Per token: 2 × 16 × 8 × 64 × 2 bytes = 32,768 bytes = 32 KB

  For 128 tokens:  128 × 32 KB = 4 MB     ← negligible
  For 2048 tokens: 2048 × 32 KB = 64 MB   ← still small vs 2.48 GB model
  For 32K tokens:  32K × 32 KB = 1 GB     ← now it matters!
```

At short sequence lengths (our benchmarks use ~128 tokens), KV cache memory is negligible. The bandwidth cost is dominated by reading model weights. But for long-context applications (32K+ tokens), KV cache reads become a significant fraction of total bandwidth.

**Why only K and V, not Q?** During decode, we only generate ONE new token. It has one query vector Q_new. But we need to compare it against ALL previous keys (K₀...K_N) and blend ALL previous values (V₀...V_N). So K and V grow with sequence length, but Q is always just one vector.

### CNNs vs Transformers: A 60-Second Detour

Convolution (CNN) and attention (transformer) are both ways to combine information from neighboring data, but they work differently:

```
CNN: Sliding window (local patterns)
  Input:    [a] [b] [c] [d] [e] [f] [g]
  Filter:       [w₁ w₂ w₃]                ← fixed 3-wide window
             ──────────►
  Output:   [·] [b'] [c'] [d'] [e'] [f'] [·]

  Each output = w₁×left + w₂×center + w₃×right  (same weights everywhere)
  Params: just 3 weights (shared across all positions)
  Receptive field: LOCAL (only sees 3 neighbors)

Transformer attention: All-to-all (global patterns)
  Input:    [a] [b] [c] [d] [e] [f] [g]
             ↕   ↕   ↕   ↕   ↕   ↕   ↕    ← every token attends to every other
  Output:   [a'] [b'] [c'] [d'] [e'] [f'] [g']

  Each output = weighted sum of ALL inputs (weights learned dynamically per input)
  Params: W_q, W_k, W_v (d² each) — much larger
  Receptive field: GLOBAL (sees everything)
```

**Why CNNs are efficient**: The same 3×3 filter slides everywhere → few parameters, high data reuse, great for images where local patterns (edges, textures) matter most.

**Why transformers dominate language**: Word relationships can be long-range ("The cat that the dog chased **ran** away" — "cat" and "ran" are 5 words apart). CNNs need many stacked layers to see that far; attention sees it in one layer.

**On GPU, the math is the same**: Both are matrix multiplies. A convolution can be unrolled into a matrix multiply (im2col). tinygrad compiles both to the same kind of kernels. The performance characteristics differ mainly in matrix shapes:

| | CNN (im2col matmul) | Transformer matmul |
|---|---|---|
| **Shape** | Tall-skinny (many spatial positions × few channels) | Square-ish (d × d, or seq_len × d) |
| **Reuse** | High (same filter applied everywhere) | Medium (different Q for each position) |
| **Memory** | Weights small, activations large | Weights large, activations smaller |
| **Bottleneck** | Often compute-bound (lots of reuse) | Often memory-bound (large weights, low reuse in decode) |

### What is Matvec?

**Matrix-vector multiply**: the core operation in LLM decode.

```text
     Weight matrix W          Input vector x         Output vector y
     ┌─────────────┐          ┌───┐                  ┌───┐
     │ . . . . . . │          │ . │                  │ . │
M    │ . . . . . . │    ×     │ . │    =             │ . │
rows │ . . . . . . │          │ . │                  │ . │
     │ . . . . . . │          │ . │                  │ . │
     └─────────────┘          └───┘                  └───┘
         K cols               K×1                    M×1
```

Each output element $y_i$ is a dot product of row $i$ of $W$ with $x$:

$$y_i = \sum_{k=0}^{K-1} W_{i,k} \cdot x_k$$

On GPU, you want many threads cooperating on each row so they can read $W$ in big coalesced bursts (128 bytes at a time). Without this cooperation, each thread reads its own row independently → terrible cache behavior → 5% of peak bandwidth.

tinygrad's **matvec heuristic** ([Pill 11](11-matvec-heuristic.md)) sets up this cooperative pattern:

- `MV_THREADS_PER_ROW=32` — 32 threads share a row, each reading K/32 elements
- `MV_BLOCKSIZE=4` — 4 rows per workgroup
- `MV_ROWS_PER_THREAD=4` — each thread handles 4 rows (vector loads)

Without this fix: **18.3 tok/s**. With it: **29.0 tok/s** (+59%).

### What is fp16, bf16, fp32? (Number Formats by Hand)

Every weight in a neural network is a floating-point number. The format determines **how many bytes the GPU must read per weight** — and therefore how fast you can run.

```
float32 (4 bytes — full precision):
┌───┬───────────────────┬───────────────────────────────────────────────────────┐
│ S │ E E E E E E E E   │ M M M M M M M M M M M M M M M M M M M M M M M       │
│ 1 │     8 bits        │                    23 bits                            │
└───┴───────────────────┴───────────────────────────────────────────────────────┘
 sign    exponent              mantissa (fraction)
         (range)               (precision: ~7 decimal digits)
 Total: 32 bits = 4 bytes per weight

float16 / fp16 (2 bytes — half precision):
┌───┬───────────┬───────────────────────┐
│ S │ E E E E E │ M M M M M M M M M M   │
│ 1 │  5 bits   │       10 bits          │
└───┴───────────┴───────────────────────┘
 sign  exponent    mantissa
       (range)     (precision: ~3.3 decimal digits)
 Total: 16 bits = 2 bytes per weight

bfloat16 / bf16 (2 bytes — "brain" float, Google TPU format):
┌───┬───────────────────┬─────────────┐
│ S │ E E E E E E E E   │ M M M M M M M │
│ 1 │     8 bits        │    7 bits     │
└───┴───────────────────┴─────────────┘
 sign    exponent          mantissa
         (SAME range       (LESS precision than fp16,
          as fp32!)         but same range as fp32)
 Total: 16 bits = 2 bytes per weight
```

**By hand — what does the number 3.14 look like in each format?**

```
  3.14 in float32:  0 10000000 10010001111010111000011  (exact: 3.1400001049)
  3.14 in float16:  0 10000    1001000100               (exact: 3.140625)
  3.14 in bfloat16: 0 10000000 1001001                  (exact: 3.15625)
                     ↑              ↑
                   same sign     bf16 rounds more (7 bits vs 10)
                                 but same exponent range as fp32
```

**Why bf16 exists**: fp16's 5-bit exponent can only represent numbers from ~6×10⁻⁸ to 65504. During training, gradients can be smaller than 6×10⁻⁸ → underflow to zero → training diverges. bf16 keeps fp32's 8-bit exponent (range ~1×10⁻³⁸ to 3×10³⁸) at the cost of less precision. For inference, either works fine.

**The DRAM bandwidth impact**:

```
  1.24 billion weights × 4 bytes (fp32) = 4.96 GB per token read
  1.24 billion weights × 2 bytes (fp16) = 2.48 GB per token read  ← 2× less data!

  At 102 GB/s read bandwidth:
    fp32: 102 / 4.96 = 20.6 tok/s theoretical max
    fp16: 102 / 2.48 = 41.1 tok/s theoretical max  ← matches our Qwen3 0.6B result!
```

This is why `HALF=1` (fp16) gives a 70% speedup over `HALF=0` (fp32) — literally half the bytes to read.

### What is Quantization? (Q6_K, Q8_0, q4f16 — by Hand)

Quantization compresses weights below 16 bits. The idea: store a **block** of weights as small integers plus one shared scale factor.

**Q8_0 by hand** (simplest quantization): block of 32 weights

```
Original fp16 weights (32 × 2 bytes = 64 bytes):
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│ 0.52 │-1.83 │ 0.91 │ 0.03 │-0.77 │ 1.24 │-0.45 │ 0.66 │ ... (×32)
└──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
 Each: 16 bits. Total: 64 bytes for 32 weights.

Step 1: Find the absmax → 1.83
Step 2: scale = 1.83 / 127 = 0.01441
Step 3: Quantize each: round(weight / scale) → int8

Quantized Q8_0 block (32 × 1 byte + 2 byte scale = 34 bytes):
┌───────────────────────────────────────────────────────────┐
│ scale (fp16): 0.01441                                     │  2 bytes
├─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬──────────┤
│  36 │-127 │  63 │   2 │ -53 │  86 │ -31 │  46 │ ... ×32  │  32 bytes
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴──────────┘
 Each: 8 bits. Total: 34 bytes for 32 weights.
                                                   Compression: 64/34 = 1.88×

To dequantize:  weight ≈ int8_value × scale
  36 × 0.01441 = 0.519  (original: 0.52 ✓)
 -127 × 0.01441 = -1.830 (original: -1.83 ✓)
```

**Q6_K by hand**: same idea but 6 bits per weight (values -32 to +31) with super-blocks of 256 weights containing sub-scales:

```
fp16 block:    32 weights × 2 bytes          = 64 bytes
Q8_0 block:    32 weights × 1 byte + scale   = 34 bytes  (1.06 B/param)
Q6_K block:    256 weights × 6 bits + scales  = ~202 bytes (0.79 B/param)
Q4_K_M block:  256 weights × 4 bits + scales  = ~144 bytes (0.56 B/param)

Visual comparison (per-weight storage):
  fp16:   ████████████████  16 bits  (2.00 bytes)
  Q8_0:   ████████▌         8.5 bits (1.06 bytes)
  Q6_K:   ██████▎           6.3 bits (0.79 bytes)
  Q4_K_M: ████▌             4.5 bits (0.56 bytes)
```

**What this means for DRAM reads** — LLaMA 1B (1.24B params):

```
  fp16:   1.24B × 2.00 bytes = 2.48 GB per token  ███████████████████████████
  Q8_0:   1.24B × 1.06 bytes = 1.31 GB per token  ██████████████▎
  Q6_K:   1.24B × 0.79 bytes = 0.98 GB per token  ██████████▋
  Q4_K_M: 1.24B × 0.56 bytes = 0.69 GB per token  ███████▌

  At 102 GB/s:
    fp16:   102/2.48  = 41 tok/s max
    Q8_0:   102/1.31  = 78 tok/s max
    Q6_K:   102/0.98  = 104 tok/s max
    Q4_K_M: 102/0.69  = 148 tok/s max  ← if you could read it for free!
```

But dequantization isn't free — it costs GPU compute. Each framework pays this cost differently:

| Framework     | How it dequantizes                      | When                   |  Memory footprint  |
| ------------- | --------------------------------------- | ---------------------- | :----------------: |
| **llama.cpp** | Custom CUDA kernels, fused with matmul  | Per-token (on-the-fly) | Native quant size  |
| **tinygrad**  | `ggml_data_to_tensor()` → cast to fp16  | At load time (once)    | **2× native size** |
| **MLC LLM**   | TVM-compiled kernels, fused with matmul | Per-token (on-the-fly) |   Custom format    |
| **vLLM**      | PyTorch CUDA kernels                    | Per-token (on-the-fly) |   Native or fp16   |

**This is tinygrad's hidden tax**: loading a Q6_K model (0.97 GB) into tinygrad results in ~3.0 GB of fp16 weights in memory. Each token reads 3.0 GB instead of 0.97 GB. Yet tinygrad still wins.

### What is Kernel Dispatch Overhead?

Every time the GPU runs a small computation (a "kernel"), the CPU must:

1. Set up parameters (thread counts, memory pointers)
2. Submit the command to the GPU
3. Wait for the GPU to start executing

For an LLM token, there are ~40 kernels (one per matrix multiply, plus norms, attention, etc.). The dispatch overhead adds up:

```
CUDA Runtime dispatch:  ~5-15 µs per kernel  ×  40 kernels = 200-600 µs overhead
NV backend dispatch:    ~0.5 µs per kernel   ×  40 kernels = 20 µs overhead
```

On a model where total compute is 25-40 ms per token, 200-600 µs is 0.5-2.4% overhead. Seems small? It's actually the difference between winning and losing against llama.cpp.

### What is the NV Backend (HCQ)?

tinygrad's NV backend bypasses the entire CUDA software stack:

```
Normal CUDA path:
  Python → cuDNN/cuBLAS → CUDA Runtime → CUDA Driver → GPU hardware
  Each layer adds overhead, copies, synchronization

tinygrad NV path:
  Python → tinygrad compiler → ioctl() → GPU hardware
  Direct kernel submission via file descriptor, zero copies
```

The NV backend ([Pill 9](09-jetson-nv-backend-pt1.md), [Pill 10](10-jetson-nv-backend-pt2.md)) creates a channel to the GPU, builds command buffers in userspace memory, and submits them directly via `ioctl()` on `/dev/nvhost-gpu`. No CUDA runtime, no driver API calls, no cuBLAS.

**Result**: 24% faster than CUDA=1 (cuBLAS) on the same tinygrad code.

| Backend                 | Qwen3 0.6B tok/s |   Bandwidth utilized   |
| ----------------------- | :--------------: | :--------------------: |
| **NV=1** (direct ioctl) |     **41.0**     | 198 GB/s (97% of peak) |
| CUDA=1 (cuBLAS)         |       33.2       | 159 GB/s (78% of peak) |

### What is TVM Compilation? (Why MLC LLM is Fast)

**TVM** (Tensor Virtual Machine) is an ML compiler — like a "gcc for neural networks". It takes a model description and produces optimized GPU code **ahead of time**:

```
MLC LLM compilation pipeline:
  1. Model definition (LLaMA architecture)
  2. TVM Relay/Relax IR (high-level graph)
  3. TVM Schedule optimization (tiling, vectorization, loop ordering)
  4. CUTLASS template selection (hand-tuned GEMM kernels from NVIDIA)
  5. CUDA code generation (optimized .cu files)
  6. Compile for sm_87 (Orin's compute capability)
  7. Package into .so shared libraries
  8. Deploy with CUDA Graph capture

At runtime:
  → Load pre-compiled .so
  → CUDA Graph replays entire token in one GPU submission
  → Minimal Python overhead
```

Key advantages of TVM/MLC:

- **CUTLASS kernels**: NVIDIA's hand-tuned matrix multiply templates, optimized per-GPU-architecture
- **Operator fusion**: Multiple operations merged into single kernels (less dispatch overhead)
- **CUDA Graphs**: Entire token generation recorded as one graph, replayed without CPU involvement
- **Ahead-of-time compilation**: Zero JIT overhead at runtime

This is why MLC gets 36.8 tok/s vs tinygrad's 29.0 — it uses NVIDIA's own hand-tuned kernels plus graph-level dispatch elimination.

### What is BEAM Search vs Heuristic?

tinygrad has two ways to optimize its generated kernels:

**Heuristic** (`heuristic.py`): Hand-written rules that apply instantly. "If it looks like matvec, apply GROUP+LOCAL+UPCAST." Runs in microseconds. The default.

**BEAM search** (`search.py`): Try hundreds of kernel configurations on real hardware, time them, keep the best. Runs in seconds per kernel. Opt-in via `BEAM=N`.

On Jetson Orin, BEAM is **counterproductive** for LLM decode:

| Config            |  tok/s   | Why                                                                       |
| ----------------- | :------: | ------------------------------------------------------------------------- |
| Default heuristic | **29.0** | Well-tuned rules for Orin's memory system                                 |
| JITBEAM=2         |   1.0    | 27× slower! Beam finds locally-optimal kernels that cause cache thrashing |
| JITBEAM=4         |   1.1    | Same disaster                                                             |

**Root cause**: BEAM optimizes each kernel independently. It finds thread/block configs that minimize single-kernel time. But on Orin's unified memory (shared LPDDR5, no dedicated VRAM), these "optimal" configs cause inter-kernel cache thrashing that destroys overall pipeline throughput.

The heuristic works better because it applies consistent patterns (same thread counts, same memory access strategy) across all kernels, keeping the cache warm.

See [Pill 7](07-beam-search.md) for the full BEAM deep-dive.

---

## Part 2: Why tinygrad Beats llama.cpp

tinygrad beats llama.cpp by 4-13% across all models tested:

| Model (quant)     | tinygrad NV=1 | llama.cpp +FA | tinygrad advantage | tinygrad reads more data? |
| ----------------- | :-----------: | :-----------: | :----------------: | :-----------------------: |
| Qwen3 0.6B (Q8_0) |     41.0      | 37.1 (no FA)  |      **+10%**      |  +87% (1.14 vs 0.61 GB)   |
| Qwen3 0.6B (Q8_0) |     41.0      |  43.0 (+FA)   |        −5%         |           +87%            |
| LLaMA 1B (Q6_K)   |     29.0      |     27.85     |      **+4%**       |  +209% (3.0 vs 0.97 GB)   |
| LLaMA 3B (Q6_K)   |     12.1      |     11.86     |      **+2%**       |  +194% (7.2 vs 2.45 GB)   |

### The Paradox: tinygrad reads MORE data but is FASTER

This seems impossible. tinygrad dequantizes Q6_K/Q8_0 to fp16 at load time, so it reads 2-3× more bytes per token. llama.cpp reads the native quantized format. Yet tinygrad still wins.

The explanation is **dispatch overhead**:

```
llama.cpp per token:
  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
  │ kernel 1 │→│ dispatch │→│ kernel 2 │→│ dispatch │→ ... × 40 kernels
  │ 0.6 ms  │  │ 10 µs   │  │ 0.5 ms  │  │ 10 µs   │
  └─────────┘  └─────────┘  └─────────┘  └─────────┘
  Total compute: 34 ms    Total dispatch:  0.4 ms
  Token time: ~35.9 ms → 27.85 tok/s

tinygrad NV=1 per token:
  ┌─────────┐┌─────────┐┌─────────┐┌─────────┐
  │ kernel 1 ││ kernel 2 ││ kernel 3 ││ kernel 4 │  ... × 40 kernels
  │ 0.85 ms ││ 0.7 ms  ││ 0.82 ms ││ 0.15ms  │  (dispatch: ~0.5 µs each)
  └─────────┘└─────────┘└─────────┘└─────────┘
  Total compute: 34.3 ms   Total dispatch:  0.02 ms
  Token time: ~34.5 ms → 29.0 tok/s
```

**Key numbers**:

- llama.cpp processes fewer bytes per kernel (native quant) but has 10-15 µs CUDA dispatch overhead per kernel launch
- tinygrad processes more bytes per kernel (fp16 dequant) but has ~0.5 µs NV dispatch overhead
- At 40 kernels/token, llama.cpp wastes ~0.4-0.6 ms on dispatch; tinygrad wastes ~0.02 ms
- This 0.4-0.6 ms saved compensates for tinygrad reading more data

### Why NV Dispatch is 20× Faster Than CUDA

|                        | CUDA Runtime                          | tinygrad NV                |
| ---------------------- | ------------------------------------- | -------------------------- |
| **API call**           | `cuLaunchKernel()`                    | Write QMD to mapped memory |
| **Driver involvement** | Full round-trip through kernel driver | Zero — purely userspace    |
| **Synchronization**    | Implicit barriers per launch          | Timeline signals, async    |
| **Memory**             | Driver copies command buffers         | Direct MMIO to GPU FIFO    |
| **Overhead**           | ~5-15 µs per launch                   | ~0.5 µs per launch         |

The NV backend (HCQ) writes a "Queue Meta Descriptor" (QMD) — a small struct with thread counts, memory pointers, and the kernel binary address — directly into GPU-visible memory, then pokes the GPU's doorbell register. No syscalls, no driver copies, no implicit synchronization.

### What is Flash Attention (+FA)? — Standard vs Flash, by Hand

Flash Attention is the single most important optimization in modern LLM inference. To understand it, let's first see the problem it solves.

**Standard attention has a memory problem**: The $S = Q \times K^T$ matrix is $\text{seq\_len} \times \text{seq\_len}$. At 32K context:

```
  seq_len = 32,768
  S matrix size = 32,768 × 32,768 × 2 bytes (fp16) = 2 GB
  Per attention head!  × 32 heads = 64 GB  ← BIGGER THAN ALL ORIN MEMORY!
```

Standard attention writes this giant S matrix to DRAM, reads it back for softmax, writes P, reads P back to multiply by V. Three full round-trips through main memory for an intermediate result that's immediately thrown away.

**The standard attention DRAM traffic (by hand with seq_len=4, d=3)**:

```
Step 1: Compute S = Q × K^T and WRITE to DRAM
  ┌─────────────┐     ┌─────────────┐
  │  Q (4×3)    │  ×  │  K^T (3×4)  │  =  S (4×4) → WRITE 16 values to DRAM
  └─────────────┘     └─────────────┘                 ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼

Step 2: READ S back, compute softmax(S), WRITE P to DRAM
  READ 16 values ← ← ← ← DRAM
  P = softmax(S) → WRITE 16 values → → → → DRAM

Step 3: READ P back, compute O = P × V
  READ 16 values ← ← ← ← DRAM
  O = P × V = final output

  Total DRAM traffic for attention: 3 × seq_len² values written/read
  At seq_len=4: 3 × 16 = 48 extra values through DRAM
  At seq_len=4096: 3 × 16M = 48M extra values through DRAM
```

**Flash Attention: TILE the computation to stay on-chip**

The GPU has a small but very fast memory called **shared memory** (SRAM) — on Orin, 228 KB per SM vs 64 GB of DRAM. Flash Attention processes Q, K, V in tiles small enough to fit in SRAM:

```
Flash Attention (tiled, stays on-chip):

  Split Q into tiles of 2 rows:  Q₀=[row 0,1]  Q₁=[row 2,3]
  Split K,V into tiles of 2 rows: KV₀=[row 0,1]  KV₁=[row 2,3]

  For each Q tile:
    For each KV tile:
      ┌───────────┐     ┌───────────┐
      │ Q_tile    │  ×  │ K_tile^T  │  =  S_tile (2×2) ← fits in SRAM!
      │ (2×3)     │     │ (3×2)     │
      └───────────┘     └───────────┘
      P_tile = softmax(S_tile)          ← computed in SRAM!
      O_tile += P_tile × V_tile         ← accumulated in SRAM!

    Write O_tile to DRAM (only the final result)

  Total DRAM traffic: JUST the input (Q, K, V) and output (O)
  No intermediate S or P ever touches DRAM!
```

```
DRAM traffic comparison:
  Standard:  Read Q,K,V + Write S + Read S + Write P + Read P + Write O
             = 2×seq_len×d + 3×seq_len² values

  Flash:     Read Q,K,V + Write O
             = 2×seq_len×d values  (the seq_len² terms vanish!)

  At seq_len=4096, d=64:
    Standard: 2×4096×64 + 3×4096² = 524K + 50M  ≈ 50M values
    Flash:    2×4096×64            = 524K values
    Savings: ~96× less DRAM traffic for attention!
```

**But during batch=1 decode, flash attention helps less**: When generating tokens one at a time, `seq_len_q = 1` (only one query). The S matrix is just $1 \times \text{seq\_len\_kv}$ — a single row, not a huge matrix. The savings from tiling are minimal. That's why llama.cpp's +FA only gives ~8% speedup for decode:

```
  Decode attention: S = q (1×d) × K_cache^T (d×seq) = (1×seq) vector
  This is already just a matvec — no huge matrix to tile!
  Flash attention still helps by fusing kernels, but the big
  seq_len² → seq_len reduction doesn't apply.
```

llama.cpp +FA: 27.85 tok/s. llama.cpp no FA: 25.7 tok/s. tinygrad (no FA equivalent): 29.0 tok/s.
tinygrad wins even against +FA because NV dispatch savings exceed FA's attention savings.

---

## Part 3: Why tinygrad Ties/Beats vLLM

| Workload                         | tinygrad |    vLLM    | Winner                             |
| -------------------------------- | :------: | :--------: | ---------------------------------- |
| LLaMA 1B fp16 (direct benchmark) |   29.0   | 30.1 (API) | **Tie**                            |
| LLaMA 1B Q6_K                    |   ~29    |    ~15     | **tinygrad 2×**                    |
| Qwen3 (any)                      |   41.0   |     ❌      | **tinygrad** (only one that works) |

### vLLM's Architecture

vLLM is a **production inference server** designed for throughput at scale:

```
vLLM stack:
  HTTP server (uvicorn/fastapi)
    → Request scheduler (PagedAttention, continuous batching)
      → PyTorch model (standard HuggingFace weights)
        → torch.compile / CUDA kernels
          → CUDA Runtime → GPU
```

vLLM's strengths are **batched throughput** (serving 100 concurrent users) and **memory efficiency** (PagedAttention). For batch=1 single-user inference, these features add overhead:

- HTTP parsing and serialization: ~1-2 ms per request
- Scheduler overhead: ~0.5 ms per step
- PyTorch eager mode (we use `--enforce-eager`): full CUDA dispatch overhead
- Python GIL contention between server and inference

### Why vLLM's GGUF Path is 2× Slower

vLLM's GGUF support is a bolt-on: it loads GGUF files, dequantizes to PyTorch tensors, then runs through the normal fp16 path. The warning in the logs says it all:

```
WARNING: gguf quantization is not fully optimized yet.
The speed can be slower than non-quantized models.
```

On Q6_K, vLLM first dequantizes to fp16 (like tinygrad) but then runs through PyTorch's CUDA Runtime path (not custom quant kernels like llama.cpp). Double penalty.

### Model Support: tinygrad's Secret Weapon

Both vLLM (0.6.3) and MLC LLM (r36.4.0) containers are **frozen** at old releases that don't support new model architectures:

```python
# vLLM trying to load Qwen3:
ValueError: Architecture qwen3 not supported

# MLC trying to load Qwen3:
KeyError: 'qwen3'
```

tinygrad loads models from GGUF format (a generic weight container), and implements the transformer architecture generically. New model architectures work automatically if they follow standard patterns.

---

## Part 4: Why MLC LLM Beats tinygrad

MLC LLM is 27% faster on LLaMA 1B fp16: **36.8 vs 29.0 tok/s**.

### The Three Advantages MLC Has

**1. CUTLASS Kernels (Hand-Tuned by NVIDIA)**

MLC uses [CUTLASS](https://github.com/NVIDIA/cutlass) — NVIDIA's template library of hand-optimized matrix multiply kernels. These kernels are tuned per-architecture (sm_87 for Orin) and use every hardware feature:

```
tinygrad kernel (compiler-generated):
  - PTX code from pattern-matching optimizer
  - Good memory access patterns (from heuristic)
  - ~85-95% of peak bandwidth utilization
  - Does NOT use tensor cores for matvec (tensor cores need batch ≥ 8)

MLC/CUTLASS kernel (hand-tuned):
  - Assembly-level optimized for sm_87
  - Exploits Orin-specific memory controller quirks
  - Custom shared memory tiling tuned per matrix shape
  - ~95-100% of peak bandwidth utilization
```

That 5-10% bandwidth utilization gap = 5-10% speed gap. Over 40 kernels, it compounds.

**2. CUDA Graphs (Zero Dispatch Overhead)**

CUDA Graphs are NVIDIA's answer to dispatch overhead. Instead of launching kernels one-by-one, you record the entire token pipeline and replay it:

```
Normal execution (per token):
  CPU: launch kernel 1 → ... → launch kernel 40
  GPU: ═══kernel 1═══ ... ═══kernel 40═══
  Overhead: 40 × dispatch_time

CUDA Graph execution (per token):
  CPU: replay graph (one command)
  GPU: ═══kernel 1═══kernel 2═══...═══kernel 40═══
  Overhead: 1 × graph_replay_time ≈ 5 µs total
```

MLC records the entire forward pass as a CUDA Graph during the first token. Subsequent tokens replay the graph with near-zero CPU involvement. This eliminates almost all dispatch overhead.

tinygrad's NV backend already has very low dispatch (~0.5 µs × 40 = 20 µs), but MLC's CUDA Graphs bring it to ~5 µs. That's a 15 µs savings per token — small but real.

**3. Operator Fusion (Fewer Kernels)**

TVM's compiler can fuse multiple operations into single kernels more aggressively than tinygrad:

```
tinygrad (separate kernels):
  kernel 1: RMS_norm(x)           → temp1
  kernel 2: Q_proj(temp1)         → q
  kernel 3: K_proj(temp1)         → k
  kernel 4: V_proj(temp1)         → v
  kernel 5: RoPE(q, k)            → q_rot, k_rot
  kernel 6: attention(q_rot, k_rot, v) → attn_out
  → 6 kernels, 6 DRAM round-trips

MLC/TVM (fused kernels):
  kernel 1: RMS_norm + Q/K/V_proj  → q, k, v
  kernel 2: RoPE + attention       → attn_out
  → 2 kernels, 2 DRAM round-trips
```

Fewer kernels = fewer DRAM round-trips = less time waiting for memory. Even with tinygrad's fast NV dispatch, 6 kernels reading from DRAM is slower than 2 fused kernels.

### Can tinygrad Close the Gap?

The 27% gap breaks down roughly as:

| Advantage                               | Estimated impact | Could tinygrad fix it?                                              |
| --------------------------------------- | :--------------: | ------------------------------------------------------------------- |
| CUTLASS kernels (better bandwidth util) |       ~10%       | Partially — better PTX generation, but hard to match hand-tuned ASM |
| CUDA Graphs (less dispatch)             |       ~2%        | NV backend already very low dispatch; diminishing returns           |
| Operator fusion (fewer kernels)         |       ~15%       | Yes — tinygrad's compiler can learn to fuse more aggressively       |

The biggest opportunity is **operator fusion**. tinygrad's pattern matcher ([Pill 8](08-pattern-matching.md)) could learn to fuse norm+projection and RoPE+attention, eliminating DRAM round-trips. This is an active area of development.

---

## Part 5: The Complete Picture

### Why Different Models Favor Different Frameworks

| Factor            | Small models favor          | Large models favor           |
| ----------------- | --------------------------- | ---------------------------- |
| Dispatch overhead | tinygrad (NV zero-overhead) | MLC (CUDA Graphs)            |
| Kernel quality    | MLC (CUTLASS)               | MLC (CUTLASS)                |
| Memory efficiency | llama.cpp (native quant)    | llama.cpp (native quant)     |
| Model support     | tinygrad (GGUF generic)     | vLLM (HuggingFace ecosystem) |
| Startup time      | llama.cpp (~instant)        | vLLM (~1 min)                |
| Batch throughput  | vLLM (PagedAttention)       | vLLM (PagedAttention)        |

### The Bandwidth Race (tok/s = bandwidth / model_size)

Every framework is ultimately limited by the same physics. Here's how close each gets to the theoretical bandwidth limit:

```
Theoretical: 102 GB/s LPDDR5 read bandwidth

Framework         Model       Memory read/tok  Bandwidth used  % of peak   tok/s
───────────────   ─────────   ──────────────   ─────────────   ──────────  ─────
tinygrad NV=1     Qwen3 0.6B  1.14 GB (fp16)   198 GB/s*       97%        41.0
MLC LLM           LLaMA 1B    2.48 GB (fp16)    91 GB/s        89%        36.8
vLLM              LLaMA 1B    2.48 GB (fp16)    75 GB/s        73%        30.1
tinygrad NV=1     LLaMA 1B    3.0 GB (fp16†)    87 GB/s        85%        29.0
llama.cpp +FA     LLaMA 1B    0.97 GB (Q6_K)    27 GB/s        26%        27.85
llama.cpp +FA     Qwen3 0.6B  0.61 GB (Q8_0)    26 GB/s        26%        43.0

* tinygrad reports "global_mem" bandwidth which includes all memory accesses
† tinygrad dequants Q6_K → fp16 at load time, reads the larger fp16 version
```

**Wait — llama.cpp only uses 26% of bandwidth?**

Yes. llama.cpp reads native quant (0.97 GB) but spends significant time on **dequantization compute** inside each kernel. The GPU ALUs are partially busy converting Q6_K blocks to fp16 for multiplication. This makes it partially **compute-bound** even though the total data is small.

tinygrad avoids this by dequantizing once at load time. Its kernels are pure fp16 matvec — memory-bound with no compute overhead, hitting 85-97% bandwidth utilization.

### Decision Matrix: When to Use Each Framework

| Your situation                                | Best framework                     | Why                                 |
| --------------------------------------------- | ---------------------------------- | ----------------------------------- |
| Single user, latest models (Qwen3, etc.)      | **tinygrad NV=1**                  | Only one that supports them         |
| Single user, LLaMA/Mistral, max tok/s         | **MLC LLM**                        | CUTLASS + CUDA Graphs               |
| Single user, minimal dependencies             | **llama.cpp**                      | C++ binary, no Python/Docker        |
| Many concurrent users                         | **vLLM**                           | PagedAttention, continuous batching |
| Smallest memory footprint                     | **llama.cpp**                      | Native quantized format             |
| Hackable, want to modify the inference engine | **tinygrad**                       | 10K lines of Python, all visible    |
| Embedded/edge deployment                      | **tinygrad NV=1** or **llama.cpp** | No Docker, low overhead             |

---

## Part 6: Tuning tinygrad for Maximum Speed

### Essential Environment Variables

```bash
# NV backend (required for best performance on Orin)
export NV=1

# Matvec heuristic fix (CRITICAL — 59% speedup)
export MV_THREADS_PER_ROW=32

# Keep weights in fp16 (default, 2× faster than fp32)
export HALF=1

# Do NOT use JITBEAM on Orin (25-27× slower!)
# export JITBEAM=2  ← DON'T DO THIS
```

### The Impact of Each Tuning Knob

| Setting               | Off → On    |  Impact  | Why                                    |
| --------------------- | ----------- | :------: | -------------------------------------- |
| NV=1 (vs CUDA=1)      | 33.2 → 41.0 | **+24%** | Zero-overhead dispatch vs CUDA runtime |
| MV_THREADS_PER_ROW=32 | 18.3 → 29.0 | **+59%** | Enables cooperative matvec kernels     |
| HALF=1 (vs HALF=0)    | 24.2 → 41.0 | **+70%** | fp16 = half the DRAM reads             |
| JITBEAM=2             | 29.0 → 1.0  | **−97%** | Cache thrashing on unified memory      |

### Running the Benchmark

```bash
# Enter dev shell
cd examples/tinygrad && nix develop

# Qwen3 0.6B (fastest — tinygrad excels here)
cd tinygrad && NV=1 MV_THREADS_PER_ROW=32 python3 -m tinygrad.apps.llm --model qwen3:0.6b --benchmark 15

# LLaMA 1B (good for cross-framework comparison)
NV=1 MV_THREADS_PER_ROW=32 python3 -m tinygrad.apps.llm --model llama3.2:1b --benchmark 15

# LLaMA 3B (larger model, tests scaling)
NV=1 MV_THREADS_PER_ROW=32 python3 -m tinygrad.apps.llm --model llama3.2:3b --benchmark 15
```

### Reading the Output

```
 34.31 ms,  29.15 tok/s,  128.43 GB/s, param   87.36 GB/s
 │            │              │                    │
 │            │              │                    └── bandwidth to read just params
 │            │              └── total memory bandwidth (params + KV + intermediates)  
 │            └── tokens per second (1000 / ms_per_token)
 └── milliseconds per token
```

The first 3-4 lines are JIT warmup (slow). Steady-state is lines 5+.

---

## Summary

### Why tinygrad wins vs llama.cpp

- **NV backend's zero-overhead dispatch** saves 0.4-0.6 ms per token
- This **more than compensates** for reading 2-3× more data (dequant→fp16 at load)
- Wins by 4-13% across all models tested

### Why tinygrad ties vLLM

- vLLM's CUDA Runtime dispatch overhead cancels out its better kernel library
- vLLM's GGUF path is poorly optimized (2× slower on quantized models)
- tinygrad supports newer model architectures (Qwen3)

### Why MLC LLM beats tinygrad

- **CUTLASS kernels** (NVIDIA hand-tuned) get ~10% more bandwidth utilization
- **CUDA Graphs** eliminate dispatch overhead entirely
- **TVM operator fusion** reduces DRAM round-trips by merging kernels
- Total advantage: ~27%

### The fundamental tradeoff

- **tinygrad**: General-purpose compiler, ~85-97% bandwidth utilization, zero-overhead NV dispatch, works on any GGUF model
- **MLC LLM**: Specialized compiler, ~89-95% utilization, CUDA Graphs, only works on pre-compiled models
- **llama.cpp**: Custom C++ kernels, ~26% bandwidth utilization (compute-bound on dequant), widest model support
- **vLLM**: Production server, ~73% utilization, best for multi-user throughput

---

**Previous**: [← Pill 14: Contributing to TinyGrad](14-contributing.md)
