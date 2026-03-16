# 04 — Scaling to 30B: Fine-Tune Guide for Jetson AGX Orin

> *This guide explains how to take the Attention Residual idea from our tiny
> experiment and apply it to a real large language model on your Jetson AGX
> Orin Dev Kit (64 GB unified memory).*

---

## Overview

The small experiment in `03_Small_Model_Experiment.py` proves the concept
on a ~500K parameter model.  Now we want to apply the same architectural
change to a production-scale model like **Llama-3.1-8B** or **Qwen2.5-32B**.

The Jetson AGX Orin's 64 GB of unified memory (shared between CPU and GPU)
makes it uniquely suited for this: models that would normally need multiple
GPUs can fit in a single device using quantization and memory-efficient
techniques.

---

## Strategy: LoRA + Attention Residual Adapters

We don't retrain the entire 30B model — that would require hundreds of GBs.
Instead, we:

1. **Freeze** the base model weights (loaded in 4-bit via quantization)
2. **Add LoRA adapters** to the attention layers (standard PEFT technique)
3. **Insert Attention Residual layers** as additional trainable modules
4. **Train only** the LoRA adapters + Attention Residual parameters

This means we train ~1–2% of the total parameters while getting the
benefit of attention-weighted residual connections.

---

## Step-by-Step: Llama-3.1-8B on Jetson Orin

> **Note:** For the fine-tuning of models at 8B+ scale, we recommend using
> PyTorch with the Hugging Face ecosystem (transformers, peft, bitsandbytes)
> since these libraries have mature quantization and LoRA support.  tinygrad
> is excellent for the core training loop and custom layers, but the
> ecosystem tooling for large pre-trained model loading is more mature in
> PyTorch.  Use tinygrad for the small experiments and custom layer
> development; use PyTorch + HF for the large-scale fine-tune.

### Step 1: Environment Setup

```bash
# On Jetson AGX Orin with JetPack 6.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install transformers peft bitsandbytes accelerate datasets
pip install trl wandb   # for training helpers and logging

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Step 2: Load the Base Model in 4-bit

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
)

model_name = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",            # auto-place on available GPU
    torch_dtype=torch.float16,
)
```

**Memory estimate:**  8B params × 4-bit ≈ **4 GB** for the frozen weights,
leaving ~60 GB for activations, gradients, and optimizer states.

### Step 3: Add LoRA Adapters

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                         # LoRA rank
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected: ~0.5% of total parameters are trainable
```

### Step 4: Insert Attention Residual Modules

```python
import torch
import torch.nn as nn

class AttentionResidualAdapter(nn.Module):
    """
    PyTorch version of AttentionResidualLayer for HF integration.
    Wraps a transformer block to add attention-weighted residuals.
    """
    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))
        self.num_heads = num_heads

    def forward(self, current, history_stack):
        # history_stack: (B, T, L, D)  where L = number of history entries
        B, T, D = current.shape
        H = self.num_heads
        Hd = D // H
        L = history_stack.shape[2]

        q = self.q_proj(current).reshape(B, T, 1, H, Hd).permute(0, 3, 1, 2, 4)
        k = self.k_proj(history_stack).reshape(B, T, L, H, Hd).permute(0, 3, 1, 2, 4)
        v = self.v_proj(history_stack).reshape(B, T, L, H, Hd).permute(0, 3, 1, 2, 4)

        scale = Hd ** -0.5
        attn = (q * scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).squeeze(3).permute(0, 2, 1, 3).reshape(B, T, D)
        out = self.out_proj(out)

        g = self.gate.sigmoid()
        return g * out + (1 - g) * current

# Insert into model (example for Llama architecture)
def inject_attention_residuals(model, block_size=4):
    """Add AttnRes adapters every block_size layers."""
    hidden_size = model.config.hidden_size
    adapters = nn.ModuleList()
    for i in range(0, len(model.model.layers), block_size):
        adapters.append(AttentionResidualAdapter(hidden_size))
    model.attn_res_adapters = adapters
    return model
```

### Step 5: Training Configuration

```python
from transformers import TrainingArguments
from trl import SFTTrainer

training_args = TrainingArguments(
    output_dir="./attnres-llama-8b",
    num_train_epochs=3,
    per_device_train_batch_size=1,       # small batch for Orin memory
    gradient_accumulation_steps=16,       # effective batch = 16
    gradient_checkpointing=True,          # saves ~40% activation memory
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    fp16=True,                            # mixed precision
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    max_grad_norm=1.0,                    # gradient clipping
    report_to="wandb",                    # or "tensorboard"
)
```

### Step 6: Dataset

```python
from datasets import load_dataset

# Good starting datasets:
# - "tatsu-lab/alpaca" (52K instruction-following examples)
# - "Open-Orca/OpenOrca" (larger, more diverse)
# - Your own JSONL file with {"text": "..."} entries

dataset = load_dataset("tatsu-lab/alpaca", split="train")

# Format for instruction fine-tuning
def format_prompt(example):
    if example.get("input"):
        return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"

dataset = dataset.map(lambda x: {"text": format_prompt(x)})
```

### Step 7: Launch Training

```bash
# On Jetson Orin — save Steps 2–6 above into a script called e.g. train_attnres_llama.py
CUDA_VISIBLE_DEVICES=0 python train_attnres_llama.py
```

---

## Memory Tips for Jetson Orin

| Technique | Memory Savings | How to Enable |
|-----------|---------------|---------------|
| 4-bit quantization (NF4) | ~75% of model weights | `BitsAndBytesConfig(load_in_4bit=True)` |
| Gradient checkpointing | ~40% of activations | `gradient_checkpointing=True` |
| LoRA (rank 16) | Train only ~0.5% of params | `peft.get_peft_model(...)` |
| Batch size 1 + grad accum | Linear memory reduction | `per_device_train_batch_size=1` |
| Mixed precision (fp16) | ~50% of compute memory | `fp16=True` |
| Block AttnRes (B=4) | 4× less history memory | `block_size=4` in adapter |

**Memory budget (approximate for 8B model on 64GB Orin):**

| Component | Memory |
|-----------|--------|
| Base model (4-bit) | ~4 GB |
| LoRA adapters | ~0.2 GB |
| AttnRes adapters | ~0.5 GB |
| Activations (checkpointed) | ~8 GB |
| Optimizer states | ~2 GB |
| **Total** | **~15 GB** |
| **Free for batch/seq** | **~49 GB** |

You have plenty of room!  You could increase batch size to 2–4 or
sequence length to 4096.

---

## How AttnRes Helps on Limited Hardware

The ~1.25× effective compute advantage means:

1. **Fewer layers needed** — a 24-layer model with AttnRes ≈ 30-layer model
   without.  Fewer layers = less memory, faster forward/backward.

2. **Better gradient utilization** — attention residuals prevent gradient
   dilution, so each training step is more effective.  You need fewer
   total steps to reach the same quality.

3. **Compound savings** — fewer layers × fewer steps = significantly less
   total compute.  On an Orin (which has great compute but limited memory
   bandwidth), this is a major win.

---

## Expected Training Times on Orin

| Model | Config | Time per Step | Total (3 epochs, Alpaca) |
|-------|--------|---------------|--------------------------|
| Llama-3.1-8B (4-bit + LoRA) | BS=1, GA=16, seq=2048 | ~3–5 sec | ~8–12 hours |
| Llama-3.1-8B + AttnRes | BS=1, GA=16, seq=2048 | ~4–6 sec | ~10–14 hours |
| Qwen2.5-32B (4-bit + LoRA) | BS=1, GA=32, seq=1024 | ~10–15 sec | ~24–36 hours |

The AttnRes variant is slightly slower per step (extra attention computation)
but typically converges in fewer steps, making total wall time similar or
better.

---

## Scaling to Qwen2.5-32B

For 32B models, use 4-bit quantization aggressively:

```python
# 32B × 4-bit ≈ 16 GB base model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-32B",
    quantization_config=bnb_config,
    device_map="auto",
)
```

With gradient checkpointing and batch size 1, this fits in 64 GB.
Add Block AttnRes (block_size=8) to keep the history memory manageable
at 64 layers.

---

## Checklist: Your Tomorrow Plan

- [ ] Install PyTorch + HF ecosystem on Jetson Orin
- [ ] Download Llama-3.1-8B weights (need HF access token)
- [ ] Run the LoRA baseline first (without AttnRes) to verify setup
- [ ] Add AttentionResidualAdapter modules
- [ ] Train for 1 epoch on Alpaca and compare loss curves
- [ ] If successful, try Qwen2.5-32B with Block AttnRes

---

## Troubleshooting

| Issue | Solution |
|-------|---------|
| OOM on Orin | Reduce `max_seq_length`, enable gradient checkpointing, use batch size 1 |
| Slow training | Enable `fp16=True`, use `torch.compile` if PyTorch 2.x is available |
| Loss not decreasing | Check learning rate (try 1e-4 to 5e-4), verify data formatting |
| bitsandbytes errors | Ensure CUDA toolkit matches PyTorch build (`nvcc --version`) |

---

Good luck!  Start with the 8B model today and scale up as you gain confidence.
The Attention Residual concept is the same at every scale — only the
engineering details change.
