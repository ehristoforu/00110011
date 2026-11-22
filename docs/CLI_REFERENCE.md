# CLI & UI Reference

## 1. CLI Commands Quick Reference

```bash
# 1. Standard YAML Merge
polymerge config.yaml -o ./merged_model

# 2. Assemble MoE from Dense models
polymerge moe_config.yaml -o ./assembled_moe --moe

# 3. Layer Slicing / Frankenmerging
polymerge slice_config.yaml -o ./frankenmerged_model --slices

# 4. Extract LoRA Adapter via SVD
polymerge --extract-lora --base /path/base --tuned /path/tuned -o ./my_lora --rank 16 --alpha 32

# 5. Bake LoRA into Base Model
polymerge --merge-lora --base /path/base --adapter ./my_lora -o ./baked_model

# 6. Merge Multiple LoRA Adapters
polymerge --merge-adapters --adapters ./math_lora ./code_lora --weights 0.6 0.4 -o ./combined_lora

# 7. Dense Layer Pruning (Local Healing)
polymerge --prune-dense --base /path/model -o ./pruned_model --drop-fraction 0.2

# 8. MoE Expert Pruning (Soft-Transfer Recovery)
polymerge --prune-moe --base /path/gemma-4 -o ./gemma_64e --target-experts 64 --alpha 0.3

# 9. Vocabulary Alignment
polymerge --align-vocab --base /path/base --models /path/m1 /path/m2 -o ./aligned_base

# 10. Model Sanity Audit (NaN/Inf & Router Degeneracy)
polymerge --audit --base /path/to/model

# 11. Auto-Quantize to GGUF
polymerge config.yaml -o ./merged_model --quantize Q4_K_M

# 12. Publish directly to Hugging Face Hub
polymerge --push-to-hub username/my-merged-model -o ./merged_model

# 13. Show Help & Guide
polymerge --help
```

---

## 2. Interactive Google Colab & Kaggle Notebook

PolyMerge includes a graphical UI notebook for Google Colab and Kaggle located at:
[`notebooks/PolyMerge_Interactive_Colab.ipynb`](../notebooks/PolyMerge_Interactive_Colab.ipynb).

### Features:
- Dropdown selection for all 9 merge methods.
- Dynamic sliders for weights, densities, and DAREx parameters.
- Checkbox to automatically ignore Vision and Multimodal projector layers.
- Real-time animated progress bars and tables via `rich`.
