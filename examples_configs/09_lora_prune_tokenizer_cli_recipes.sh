# PolyMerge Full CLI & Tool Recipes
# Architecture-Agnostic, Pure Streaming, Multi-Device LLM Toolkit

# ==========================================
# 1. Vocabulary Surgery (Alignment & Truncation)
# ==========================================
# Align and expand embedding tensors to union vocabulary, padded to 64x for Tensor Cores:
polymerge --align-vocab --base meta-llama/Llama-3.2-1B --models custom-instruct-model -o ./aligned_base --pad-multiple 64

# Truncate bloated vocabulary back to exact count:
polymerge --truncate-vocab 128000 --base ./aligned_base -o ./truncated_base

# ==========================================
# 2. Low-Rank Adaptation (LoRA) Toolkit
# ==========================================
# Extract adapter delta via Truncated SVD from base vs fine-tuned weights:
polymerge --extract-lora --base meta-llama/Llama-3.2-1B --tuned ./finetuned_model -o ./extracted_lora --rank 16 --alpha 32

# Mathematically bake adapter into base model Safetensors without backprop:
polymerge --merge-lora --base meta-llama/Llama-3.2-1B --adapter ./extracted_lora -o ./baked_model

# Merge multiple LoRA adapters (Linear or DARE-TIES):
polymerge --merge-adapters --adapters ./lora_code ./lora_math --weights 0.6 0.4 --method dare_ties --density 0.7 -o ./merged_lora

# ==========================================
# 3. Zero-Retraining Compression & Pruning
# ==========================================
# Dense Layer Pruning via Cosine Block Redundancy & Local Residual Healing:
polymerge --prune-dense --base ./baked_model -o ./pruned_dense_model --drop-fraction 0.2

# MoE Expert Pruning with Soft-Transfer Router Recovery (T18 framework):
polymerge --prune-moe --base google/gemma-4-26B-A4B-it -o ./gemma-64e --target-experts 64 --alpha 0.3

# ==========================================
# 4. Model Sanity Audit & Integrity Verification
# ==========================================
# Audit model tensors for NaN/Inf values and MoE router matrix condition number:
polymerge --audit --base ./pruned_dense_model

# ==========================================
# 5. GGUF Quantization & Hugging Face Hub Publication
# ==========================================
# Merge with automatic GGUF quantization (Q4_K_M):
polymerge examples_configs/04_della_magprune_qwen.yaml -o ./quantized_output --quantize Q4_K_M

# Push completed merged model directly to Hugging Face Hub with generated card:
polymerge --push-to-hub username/polymerge-custom-7b --base ./pruned_dense_model
