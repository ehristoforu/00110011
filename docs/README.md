# PolyMerge Documentation Sitemap

Welcome to the comprehensive documentation for **PolyMerge**, the high-performance, architecture-agnostic Large Language Model merger, upcycler, and pruner.

## Documentation Index

1. [**Core Merging & Methods**](MERGE_METHODS.md)
   - Mathematical descriptions of all 12 merge methods (`linear`, `slerp`, `nuslerp`, `karcher`, `nearswap`, `task_arithmetic`, `ties`, `dare_ties`, `della`, `omnimerge_v2`, `model_stock`, `sce`).
   - Pure Safetensors streaming & GGUF quantized merging.
   - Comprehensive YAML config specifications and copy-paste examples.

2. [**MoE Upcycling & Architectures**](MOE_UPCYCLING.md)
   - Assembling sparse MoE from dense experts.
   - Support for Mixtral, Qwen2.5/3.5 MoE, DeepSeek-V2/V3, and Gemma 4 MoE (`26B-A4B`).
   - Shared experts (`shared_expert`) and routing scalers.

3. [**Model Pruning & Compression**](PRUNING.md)
   - Dense layer pruning via Cosine Block Similarity & Local Healing.
   - MoE expert pruning via L2 Norm Profiling and Soft-Transfer Router Recovery (T18 framework).
   - Zero-backprop fast execution without retraining.

4. [**Layer Slicing & Frankenmerging**](LAYER_SLICING.md)
   - Arbitrary layer stacking, duplication, and re-indexing (e.g. 32L -> 48L).
   - Auto-updating `config.json` `num_hidden_layers`.

5. [**LoRA Tools**](LORA_TOOLS.md)
   - LoRA extraction from weight deltas via Truncated SVD.
   - Permanent baking of LoRA adapters into base Safetensors models.
   - Multi-adapter merging (linear and DARE-TIES).

6. [**Tokenizer & Vocabulary Alignment**](TOKENIZER.md)
   - Aligning mismatched vocabulary dimensions across models.
   - Vocabulary truncation and GPU tensor padding.

7. [**Hardware, Memory & Staged Optimization**](HARDWARE_OPTIMIZATION.md)
   - Multi-device routing (`cpu`, `cuda`, `tpu`).
   - Staged merging pipeline for low-RAM domestic computers.
   - Accelerated Hugging Face downloading with `hf-transfer` and `hf-xet`.

8. [**CLI & UI Reference**](CLI_REFERENCE.md)
   - Complete command-line manual, flags, and Google Colab / Kaggle widget guide.
