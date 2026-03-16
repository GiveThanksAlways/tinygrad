#!/usr/bin/env python3
"""
03_Small_Model_Experiment.py  —  Attention Residuals in tinygrad
================================================================

A complete, runnable training script that trains two *tiny* transformer
language models side-by-side:

  1. **ClassicTransformer**   — standard  x = x + sublayer(x)  residuals
  2. **AttnResTransformer**   — Attention-Residual aggregation (paper idea)

Both models share the same hyper-parameters (depth, width, heads) and are
trained on the **same data** (a simple next-token-prediction task on digit
sequences) so you can directly compare their loss curves.

Run
---
    cd <repo-root>
    python examples/attention_residuals/03_Small_Model_Experiment.py

On a Jetson AGX Orin you can set  DEVICE=CUDA  (the default for tinygrad
when CUDA is available).  On a laptop  DEVICE=CPU  works fine for this
tiny model.

What you will learn
-------------------
* How a training loop works (forward → loss → backward → optimizer step).
* How residual connections propagate gradients through deep networks.
* Why replacing uniform residuals with *learned attention* over history
  can lower the loss faster — the core claim of the Attention Residuals
  concept.

Every section below is heavily commented so you can read the code like a
tutorial.
"""

# ── imports ─────────────────────────────────────────────────────────────
from tinygrad import Tensor, nn, GlobalCounters, Device
from tinygrad.nn.state import get_parameters
from tinygrad.helpers import trange, getenv

# Our custom Attention-Residual layer (see utils.py for the full explanation)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import AttentionResidualLayer

# ── hyper-parameters ────────────────────────────────────────────────────
# These are deliberately small so the experiment finishes in minutes on
# any machine.  On a Jetson Orin you can bump them up.
VOCAB       = 16        # tiny vocabulary (digits 0-15)
SEQ_LEN     = 32        # sequence length
EMBED_DIM   = 128       # model width  (d_model)
NUM_HEADS   = 4         # attention heads
FF_DIM      = 256       # feed-forward hidden size
NUM_LAYERS  = 6         # transformer depth
BATCH       = getenv("BS", 64)
STEPS       = getenv("STEPS", 300)
LR          = 3e-4
SEED        = 42

print(f"[config] device={Device.DEFAULT} layers={NUM_LAYERS} embed={EMBED_DIM} "
      f"heads={NUM_HEADS} ff={FF_DIM} bs={BATCH} steps={STEPS}")

# ── synthetic data ──────────────────────────────────────────────────────
# We create a trivial "copy-shift" task: given a random sequence of tokens
# the target is the same sequence shifted by one position.  This forces the
# model to learn identity-like mappings — a setting where residual quality
# matters a lot because the gradient signal is thin.

Tensor.manual_seed(SEED)

def make_batch() -> tuple[Tensor, Tensor]:
  """Return (input, target) each of shape (BATCH, SEQ_LEN)."""
  data = Tensor.randint(BATCH, SEQ_LEN + 1, high=VOCAB)
  return data[:, :-1].contiguous(), data[:, 1:].contiguous()

# ── model components ────────────────────────────────────────────────────

class FeedForward:
  """Simple two-layer MLP with GELU, used in every transformer block."""
  def __init__(self, embed_dim: int, ff_dim: int):
    self.w1 = nn.Linear(embed_dim, ff_dim)
    self.w2 = nn.Linear(ff_dim, embed_dim)
  def __call__(self, x: Tensor) -> Tensor:
    return self.w2(self.w1(x).gelu())

class ClassicBlock:
  """
  A standard pre-norm transformer block with *uniform* residuals:

      x = x + Attn(LN(x))
      x = x + FFN(LN(x))

  This is the baseline every modern LLM uses today.
  """
  def __init__(self, embed_dim: int, num_heads: int, ff_dim: int):
    self.num_heads = num_heads
    self.head_dim = embed_dim // num_heads
    self.ln1 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))
    self.ln2 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))
    self.q = nn.Linear(embed_dim, embed_dim)
    self.k = nn.Linear(embed_dim, embed_dim)
    self.v = nn.Linear(embed_dim, embed_dim)
    self.o = nn.Linear(embed_dim, embed_dim)
    self.ff = FeedForward(embed_dim, ff_dim)

  def __call__(self, x: Tensor) -> Tensor:
    B, T, D = x.shape
    # --- self-attention with pre-norm ---
    h = x.layernorm().linear(*self.ln1)
    q = self.q(h).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
    k = self.k(h).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
    v = self.v(h).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
    attn = Tensor.scaled_dot_product_attention(q, k, v, is_causal=True)
    attn = attn.transpose(1, 2).reshape(B, T, D)
    attn = self.o(attn)
    x = x + attn                          # ← uniform residual
    # --- feed-forward with pre-norm ---
    x = x + self.ff(x.layernorm().linear(*self.ln2))   # ← uniform residual
    return x

class ClassicTransformer:
  """Decoder-only transformer with uniform residuals (the baseline)."""
  def __init__(self):
    self.tok_emb = nn.Embedding(VOCAB, EMBED_DIM)
    self.pos_emb = nn.Embedding(SEQ_LEN, EMBED_DIM)
    self.blocks = [ClassicBlock(EMBED_DIM, NUM_HEADS, FF_DIM) for _ in range(NUM_LAYERS)]
    self.ln_f = (Tensor.ones(EMBED_DIM), Tensor.zeros(EMBED_DIM))
    self.head = nn.Linear(EMBED_DIM, VOCAB)

  def __call__(self, idx: Tensor) -> Tensor:
    B, T = idx.shape
    pos = Tensor.arange(T)
    x = self.tok_emb(idx) + self.pos_emb(pos)
    for block in self.blocks:
      x = block(x)                         # uniform residual inside
    x = x.layernorm().linear(*self.ln_f)
    return self.head(x)                     # (B, T, VOCAB) logits


# ── Attention-Residual transformer ──────────────────────────────────────

class AttnResBlock:
  """
  Transformer block that returns its *raw* output (before the residual)
  so the AttentionResidualLayer can aggregate history.

  The residual connection is handled *externally* by the model's forward
  loop rather than inside the block — this is the key structural change.
  """
  def __init__(self, embed_dim: int, num_heads: int, ff_dim: int):
    self.num_heads = num_heads
    self.head_dim = embed_dim // num_heads
    self.ln1 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))
    self.ln2 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))
    self.q = nn.Linear(embed_dim, embed_dim)
    self.k = nn.Linear(embed_dim, embed_dim)
    self.v = nn.Linear(embed_dim, embed_dim)
    self.o = nn.Linear(embed_dim, embed_dim)
    self.ff = FeedForward(embed_dim, ff_dim)

  def __call__(self, x: Tensor) -> Tensor:
    B, T, D = x.shape
    h = x.layernorm().linear(*self.ln1)
    q = self.q(h).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
    k = self.k(h).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
    v = self.v(h).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
    attn = Tensor.scaled_dot_product_attention(q, k, v, is_causal=True)
    attn = attn.transpose(1, 2).reshape(B, T, D)
    attn = self.o(attn)
    # NOTE: we do NOT add the residual here — the caller handles it
    h = (attn + x).layernorm().linear(*self.ln2)
    out = attn + self.ff(h)
    return out

class AttnResTransformer:
  """
  Decoder-only transformer with *Attention Residuals*.

  Instead of  x = x + block(x)  we maintain a *history* list of every
  layer's output, and use a lightweight AttentionResidualLayer to compute
  the aggregated hidden state passed to the next layer.
  """
  def __init__(self):
    self.tok_emb = nn.Embedding(VOCAB, EMBED_DIM)
    self.pos_emb = nn.Embedding(SEQ_LEN, EMBED_DIM)
    self.blocks = [AttnResBlock(EMBED_DIM, NUM_HEADS, FF_DIM) for _ in range(NUM_LAYERS)]
    # One AttentionResidualLayer per transformer block
    self.attn_res = [AttentionResidualLayer(EMBED_DIM, num_heads=2) for _ in range(NUM_LAYERS)]
    self.ln_f = (Tensor.ones(EMBED_DIM), Tensor.zeros(EMBED_DIM))
    self.head = nn.Linear(EMBED_DIM, VOCAB)

  def __call__(self, idx: Tensor) -> Tensor:
    B, T = idx.shape
    pos = Tensor.arange(T)
    x = self.tok_emb(idx) + self.pos_emb(pos)

    # History tracks every layer's output for the attention residual
    history: list[Tensor] = [x]

    for block, ar in zip(self.blocks, self.attn_res):
      raw = block(x)                        # raw sub-layer output
      x, history = ar(raw, history)          # attention-weighted residual

    x = x.layernorm().linear(*self.ln_f)
    return self.head(x)

# ── training loop ───────────────────────────────────────────────────────

def train_model(name: str, model, steps: int = STEPS) -> list[float]:
  """
  Train `model` for `steps` gradient updates and return the loss curve.

  This is the heart of neural network training:
    1. Sample a batch of data
    2. Forward pass → compute predictions
    3. Compute loss (cross-entropy for language modelling)
    4. Backward pass → compute gradients
    5. Optimizer step → update weights
    6. Zero gradients → prepare for next step
  """
  params = get_parameters(model)
  print(f"\n{'='*60}")
  print(f"  Training: {name}")
  print(f"  Parameters: {sum(p.numel() for p in params):,}")
  print(f"{'='*60}")

  opt = nn.optim.Adam(params, lr=LR)
  losses: list[float] = []

  Tensor.training = True
  for step in (t := trange(steps)):
    GlobalCounters.reset()

    # 1. Get a batch
    x, y = make_batch()

    # 2-3. Forward + loss
    logits = model(x)                                # (B, T, VOCAB)
    loss = logits.flatten(0, 1).sparse_categorical_crossentropy(y.flatten())

    # 4. Backward — this is where the magic happens!
    #    tinygrad builds a computation graph during the forward pass and
    #    now traverses it in reverse to compute ∂loss/∂param for every
    #    parameter.  Attention Residuals change *how* the gradient flows
    #    through the residual connections.
    opt.zero_grad()
    loss.backward()

    # 5. Optimizer step — AdamW adjusts each parameter using its gradient
    #    plus running estimates of the first and second moments.
    opt.step()

    loss_val = loss.item()
    losses.append(loss_val)
    t.set_description(f"{name} loss={loss_val:.4f}")

  Tensor.training = False
  return losses

# ── main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
  print("╔══════════════════════════════════════════════════════════════╗")
  print("║  Attention Residuals — Small Model Experiment (tinygrad)    ║")
  print("╠══════════════════════════════════════════════════════════════╣")
  print("║  Comparing uniform residuals vs. attention residuals on a   ║")
  print("║  tiny next-token-prediction task.                           ║")
  print("╚══════════════════════════════════════════════════════════════╝")

  # --- Train the baseline (classic residuals) ---
  classic_model = ClassicTransformer()
  classic_losses = train_model("Classic", classic_model)

  # --- Train the attention-residual variant ---
  Tensor.manual_seed(SEED)  # reset seed for fair comparison
  attnres_model = AttnResTransformer()
  attnres_losses = train_model("AttnRes", attnres_model)

  # --- Print summary ---
  print("\n" + "="*60)
  print("  RESULTS SUMMARY")
  print("="*60)
  c_final = sum(classic_losses[-20:]) / 20
  a_final = sum(attnres_losses[-20:]) / 20
  print(f"  Classic  final avg loss (last 20 steps): {c_final:.4f}")
  print(f"  AttnRes  final avg loss (last 20 steps): {a_final:.4f}")
  if a_final < c_final:
    pct = (1 - a_final / c_final) * 100
    print(f"  → Attention Residuals improved loss by {pct:.1f}%")
  else:
    print(f"  → Classic model had lower loss this run (try more steps!)")
  print("="*60)

  # --- Optional: save loss curves to CSV for plotting ---
  try:
    with open("/tmp/attnres_losses.csv", "w") as f:
      f.write("step,classic,attnres\n")
      for i, (c, a) in enumerate(zip(classic_losses, attnres_losses)):
        f.write(f"{i},{c:.6f},{a:.6f}\n")
    print("\nLoss curves saved to /tmp/attnres_losses.csv")
    print("Plot with:  python -c \"")
    print("  import matplotlib.pyplot as plt, csv")
    print("  data = list(csv.DictReader(open('/tmp/attnres_losses.csv')))")
    print("  plt.plot([float(r['classic']) for r in data], label='Classic')")
    print("  plt.plot([float(r['attnres']) for r in data], label='AttnRes')")
    print("  plt.legend(); plt.xlabel('Step'); plt.ylabel('Loss')")
    print("  plt.title('Uniform vs Attention Residuals')")
    print("  plt.savefig('/tmp/attnres_curves.png'); plt.show()\"")
  except Exception:
    pass
