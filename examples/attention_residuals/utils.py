"""
Attention Residuals — utility layers implemented in tinygrad.

The core idea (from the Moonshot AI "Attention Residuals" concept):
  Instead of the fixed uniform residual  x_{l+1} = x_l + F_l(x_l),
  we compute an *attention-weighted* combination of ALL preceding
  hidden states so that each layer can selectively read from the
  history rather than blindly accumulating it.

This file provides two drop-in replacements for the standard residual:

  1. AttentionResidualLayer  – full per-layer attention over history
     (best quality, O(L²) memory in the number of layers L)

  2. BlockAttentionResidual  – "Block AttnRes" compression trick:
     group every `block_size` layers and keep one summary per block.
     Reduces memory from O(L²) to O(L/B · B) ≈ O(L).
"""

from tinygrad import Tensor
from tinygrad.nn import Linear

# ---------------------------------------------------------------------------
# 1.  AttentionResidualLayer
# ---------------------------------------------------------------------------

class AttentionResidualLayer:
  """
  Replaces  ``x = x + sublayer(x)``  with a learned attention gate that
  looks at *all* preceding layer outputs and produces an aggregated
  residual via multi-head attention.

  Forward signature
  -----------------
  __call__(current, history) -> (output, updated_history)

      current  : Tensor (B, T, D)  — output of the current sub-layer
      history  : list[Tensor]      — outputs of all preceding layers
                                      each entry is (B, T, D)

  Returns the attention-aggregated hidden state and the history list
  with `current` appended.

  Why this matters
  ----------------
  Uniform residuals suffer from *hidden-state dilution*: as the network
  gets deeper the earliest layers' contributions shrink by 1/L.  Attention
  Residuals let the network *choose* which earlier representations matter,
  effectively giving a 1.25× compute advantage at the same depth.
  """

  def __init__(self, embed_dim: int, num_heads: int = 4):
    assert embed_dim % num_heads == 0
    self.num_heads = num_heads
    self.head_dim = embed_dim // num_heads
    self.embed_dim = embed_dim

    # Lightweight projections — Q comes from the current layer,
    # K/V come from the history stack.
    self.q_proj = Linear(embed_dim, embed_dim)
    self.k_proj = Linear(embed_dim, embed_dim)
    self.v_proj = Linear(embed_dim, embed_dim)
    self.out_proj = Linear(embed_dim, embed_dim)

    # A learned scalar gate so the model can smoothly interpolate
    # between "pure attention residual" and "classic residual".
    self.gate = Tensor.zeros(1)  # sigmoid(0)=0.5 → equal mix at init

  def __call__(self, current: Tensor, history: list[Tensor]) -> tuple[Tensor, list[Tensor]]:
    # history contains all previous layer outputs: [h_0, h_1, ..., h_{l-1}]
    # current is h_l (the raw output of layer l, before the residual)
    B, T, D = current.shape

    # Stack history into a single tensor: (B, T, L, D) where L = num previous layers
    if len(history) == 0:
      # First layer — no history, fall back to classic residual
      return current, [current]

    # (L, B, T, D) -> (B, T, L, D) after stack and transpose
    hist_stack = Tensor.stack(*history, dim=0)          # (L, B, T, D)
    hist_stack = hist_stack.permute(1, 2, 0, 3)         # (B, T, L, D)

    # Query from current layer output
    q = self.q_proj(current)                             # (B, T, D)
    q = q.reshape(B, T, 1, self.num_heads, self.head_dim)
    q = q.permute(0, 3, 1, 2, 4)                        # (B, H, T, 1, Hd)

    # Key, Value from history
    k = self.k_proj(hist_stack)                          # (B, T, L, D)
    v = self.v_proj(hist_stack)                          # (B, T, L, D)
    L = k.shape[2]
    k = k.reshape(B, T, L, self.num_heads, self.head_dim)
    k = k.permute(0, 3, 1, 2, 4)                        # (B, H, T, L, Hd)
    v = v.reshape(B, T, L, self.num_heads, self.head_dim)
    v = v.permute(0, 3, 1, 2, 4)                        # (B, H, T, L, Hd)

    # Scaled dot-product attention over the history dimension
    scale = float(self.head_dim) ** -0.5
    attn_weights = (q * scale).matmul(k.transpose(-2, -1))  # (B, H, T, 1, L)
    attn_weights = attn_weights.softmax()                     # softmax over L
    attn_out = attn_weights.matmul(v)                         # (B, H, T, 1, Hd)

    # Merge heads back
    attn_out = attn_out.squeeze(3)                       # (B, H, T, Hd)
    attn_out = attn_out.permute(0, 2, 1, 3)             # (B, T, H, Hd)
    attn_out = attn_out.reshape(B, T, D)                 # (B, T, D)
    attn_out = self.out_proj(attn_out)

    # Gated combination: gate * attn_residual + (1 - gate) * classic_residual
    g = self.gate.sigmoid()
    # Classic residual uses most recent history entry
    classic = history[-1] + current
    output = g * attn_out + (1 - g) * classic

    new_history = history + [output]
    return output, new_history


# ---------------------------------------------------------------------------
# 2.  BlockAttentionResidual — compression trick for deep models
# ---------------------------------------------------------------------------

class BlockAttentionResidual:
  """
  "Block AttnRes" — instead of attending over every single previous layer,
  we group layers into blocks of `block_size` and keep one compressed
  summary vector per block (the mean of the block's outputs).

  This reduces the attention context from L entries to ≈ L/block_size,
  making it practical even for 96-layer models.

  Usage is identical to AttentionResidualLayer.
  """

  def __init__(self, embed_dim: int, num_heads: int = 4, block_size: int = 4):
    self.block_size = block_size
    self.attn_res = AttentionResidualLayer(embed_dim, num_heads)

  def __call__(self, current: Tensor, history: list[Tensor]) -> tuple[Tensor, list[Tensor]]:
    # Compress history into block summaries
    compressed: list[Tensor] = []
    for i in range(0, len(history), self.block_size):
      block = history[i:i + self.block_size]
      # Mean-pool the block into a single summary
      block_stack = Tensor.stack(*block, dim=0)  # (block_len, B, T, D)
      compressed.append(block_stack.mean(axis=0))  # (B, T, D)

    # Run the standard attention residual on the compressed history
    output, _ = self.attn_res(current, compressed)

    # Return full (uncompressed) history for downstream layers
    new_history = history + [output]
    return output, new_history
