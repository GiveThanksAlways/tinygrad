# Attention-Residuals-Orin-Lab

An educational project exploring the **Attention Residuals** concept — a method that replaces
the fixed uniform residual connection (`output = layer(x) + x`) found in every modern
transformer with a *learned, input-dependent attention mechanism* over all preceding layer
outputs.  The core insight is that uniform residuals cause **hidden-state dilution**: as
depth increases, the contribution of early layers shrinks as 1/L, wasting capacity.  By
letting each layer attend to the full history of hidden states, the network can selectively
amplify or suppress earlier representations, yielding an effective **~1.25× compute
advantage** at the same parameter count and depth.  This project implements the idea from
scratch in **tinygrad** — a minimalist tensor library — so you can see every moving part.

---

## Hardware Requirements

| Component | Recommendation |
|-----------|---------------|
| **GPU** | NVIDIA Jetson AGX Orin (64 GB unified memory) — or any CUDA GPU |
| **JetPack** | 6.x (CUDA 12.x) |
| **Python** | 3.10+ |
| **Framework** | tinygrad (installed from this repo) |

> **This project runs on CPU too!**  The small experiment (script 03) finishes in
> minutes on a laptop.  The Jetson-specific tips in guide 04 are for scaling up.

---

## Quick Setup

```bash
# 1. Clone and install tinygrad (you are already inside the repo)
pip install -e .

# 2. Install optional helpers
pip install numpy tqdm matplotlib

# 3. (Jetson only) Make sure CUDA works
python -c "from tinygrad import Device; print(Device.DEFAULT)"
# Should print  CUDA  or  NV

# 4. Run the small experiment TODAY
python examples/attention_residuals/03_Small_Model_Experiment.py

# 5. Read the tutorials at your own pace
#    examples/attention_residuals/01_Training_Basics.md
#    examples/attention_residuals/02_Attention_Residuals_Explained.md
#    examples/attention_residuals/04_Scale_to_30B_FineTune_Guide.md
```

---

## Project Structure

```
examples/attention_residuals/
├── README.md                              ← you are here
├── 01_Training_Basics.md                  ← tutorial: SGD, residuals, gradients
├── 02_Attention_Residuals_Explained.md    ← deep dive into the paper's math
├── 03_Small_Model_Experiment.py           ← runnable training script (tinygrad)
├── 04_Scale_to_30B_FineTune_Guide.md      ← scaling to 30B on Jetson Orin
├── utils.py                               ← AttentionResidualLayer + BlockAttnRes
├── requirements.txt                       ← pip dependencies
└── .gitignore
```

---

## How to Read This Project

1. **Start with `01_Training_Basics.md`** — understand forward/backward passes,
   residual connections, and hidden-state dilution.
2. **Read `02_Attention_Residuals_Explained.md`** — see the math and code for the
   Attention Residual mechanism, including the Block AttnRes compression trick.
3. **Run `03_Small_Model_Experiment.py`** — watch two tiny transformers train
   side-by-side (classic vs. attention residuals) and compare their loss curves.
4. **Plan your scale-up with `04_Scale_to_30B_FineTune_Guide.md`** — learn how to
   apply the same idea to a real 8B–32B model on your Jetson Orin.

---

## Environment Variables

tinygrad is controlled via environment variables:

| Variable | Effect |
|----------|--------|
| `DEBUG=1..7` | Increasing verbosity (2 shows kernel timing) |
| `DEVICE=CPU` | Force CPU execution |
| `DEVICE=CUDA` | Force CUDA execution |
| `BS=128` | Override batch size in the training script |
| `STEPS=500` | Override number of training steps |

Example:

```bash
DEBUG=2 STEPS=100 python examples/attention_residuals/03_Small_Model_Experiment.py
```

---

## License

This educational project is part of the tinygrad repository and follows its license.
