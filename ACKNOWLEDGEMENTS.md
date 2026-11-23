# Acknowledgements & Credits

PolyMerge stands on the shoulders of giants in the open-source Large Language Model and model-merging ecosystem. We express our deepest gratitude and credit to the following foundational projects, research efforts, and teams:

---

### 1. [OmniMergeKit](https://github.com/mann1x/omnimergekit) by mann1x
- **Contribution:** Pioneered the pure tensor-level merge streaming paradigm without dependence on rigid Hugging Face modeling classes. Introduced **Omnimerge v2** (OBIM top-k masking, DAREx $1/q$ rescaling, and EMR election), Soft-Transfer Router Recovery (T18 framework), and REAP expert pruning.
- **License:** MIT License.

### 2. [MergeKit](https://github.com/arcee-ai/mergekit) by Arcee AI
- **Contribution:** The gold-standard pioneer of LLM model merging in the open-source community. Established standardized YAML merge specifications, Task Arithmetic, TIES, DARE, Model Stock, and initial Frankenmerge / Layer Slicing ideas.
- **License:** GNU Lesser General Public License v3.0 (LGPL-3.0).

### 3. [MergeKitty](https://github.com/allura-org/mergekitty) by Allura
- **Contribution:** Advanced extensions of MergeKit with enhanced LoRA merging, LayerShuffle, and tokensurgeon utilities.
- **License:** GNU Lesser General Public License v3.0 (LGPL-3.0).

### 4. [Mergoo](https://github.com/Leeroo-AI/mergoo) by Leeroo AI
- **Contribution:** Pioneered early open-weights MoE expert composition (Llama/Mistral MoE assembly) and safe saving mechanisms for multi-expert checkpoints.
- **License:** GNU Lesser General Public License v3.0 (LGPL-3.0).

### 5. [MergeUI](https://github.com/naskio/mergeui) by Mehdi Nassim Khodja (naskio)
- **Contribution:** Web-based UI interfaces and graph visualization paradigms for model merging workflows.
- **License:** Apache License 2.0.

### 6. [PruneMe](https://github.com/arcee-ai/PruneMe) by Arcee AI
- **Contribution:** Empirical layer pruning strategies for LLMs based on block similarity metrics without backpropagation.
- **License:** Apache License 2.0.

---

### Research Papers & Methods Cited
- **DARE:** *Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch* (Yu et al., 2023).
- **TIES-Merging:** *Resolving Interference When Merging Models* (Yadav et al., 2023).
- **Model Stock:** *Model Stock: All We Need Is just a Few Fine-Tuned Models* (Jang et al., 2024).
- **DELLA:** *DELLA-Merging: Reducing Interference in Model Merging through Magnitude-Proportional Trimming* (2024).
- **DeepSeekMoE & Mixtral:** Architectural foundations for fine-grained routing and shared expert isolation.
