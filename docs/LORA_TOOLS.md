# LoRA Tools Guide

PolyMerge provides a standalone pure-tensor LoRA toolkit implemented in `polymerge/lora.py`. It operates without dependencies on heavy training libraries.

---

## 1. LoRA Extraction from Weight Deltas (`--extract-lora`)

Given a fine-tuned model $W_{\text{tuned}}$ and its base model $W_{\text{base}}$, this tool extracts a low-rank adapter through Truncated Singular Value Decomposition (SVD):
$$\Delta W = W_{\text{tuned}} - W_{\text{base}} \approx U_r S_r V_r^T$$

The matrices are factored into standard LoRA matrices:
$$B = \frac{U_r \sqrt{S_r}}{\alpha / r}, \quad A = \sqrt{S_r} V_r^T$$
where $\Delta W = B \cdot A \cdot \frac{\alpha}{r}$.

### Command
```bash
polymerge --extract-lora \
    --base meta-llama/Llama-3.2-1B \
    --tuned my-finetuned-llama \
    -o ./my_extracted_lora \
    --rank 16 \
    --alpha 32.0
```
- **Output:** Standard Hugging Face PEFT format:
  - `adapter_model.safetensors`
  - `adapter_config.json`

---

## 2. LoRA Merging & Baking (`--merge-lora`)

Bakes any Hugging Face PEFT LoRA adapter directly into the base model's Safetensors checkpoints:
$$W_{\text{merged}} = W_{\text{base}} + \frac{\alpha}{r} (B \cdot A)$$

### Command
```bash
polymerge --merge-lora \
    --base meta-llama/Llama-3.2-1B \
    --adapter ./my_extracted_lora \
    -o ./baked_model
```

---

## 3. Merging Multiple LoRA Adapters (`--merge-adapters`)

Merges $N$ distinct LoRA adapters together into a single combined adapter without altering the base model:

### Methods Supported:
- `linear`: Weighted blend of adapter matrices.
- `dare_ties`: Bernoulli drop and TIES sign consensus across adapter parameters.

### Command
```bash
polymerge --merge-adapters \
    --adapters ./coding_lora ./math_lora \
    --weights 0.6 0.4 \
    --method dare_ties \
    --density 0.7 \
    -o ./combined_lora
```
