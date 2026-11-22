# Tokenizer & Vocabulary Alignment Guide

When merging models trained with different tokenizers or custom added tokens (e.g., base model with 32,000 vocab and a fine-tune with 32,256 vocab), parameter tensors for `embed_tokens` and `lm_head` will have mismatched dimensions.

`TokenizerTool` resolves these discrepancies prior to merging.

---

## 1. Aligning Vocabularies (`--align-vocab`)

Finds the maximum vocabulary size among all models, pads smaller embedding matrices with zeros along dimension 0, and aligns the vocabulary size to a multiple of $N$ (e.g. 64) to optimize NVIDIA Tensor Core throughput:

### Command
```bash
polymerge --align-vocab \
    --base meta-llama/Llama-3.2-1B \
    --models meta-llama/Llama-3.2-1B-Instruct custom-finetune \
    -o ./aligned_base \
    --pad-multiple 64
```
- **Tensors Resized:**
  - `model.embed_tokens.weight`
  - `lm_head.weight`
  - `transformer.wte.weight`
- **Metadata Updated:**
  - `vocab_size` in `config.json`.
  - Tokenizer JSON definitions and model files copied over.

---

## 2. Vocabulary Truncation (`--truncate-vocab`)

Safely truncates bloated vocabulary matrices down to a specific size:

### Command
```bash
polymerge --truncate-vocab 128000 \
    --base /path/to/bloated_model \
    -o ./truncated_model
```
