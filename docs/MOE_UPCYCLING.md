# MoE Upcycling & Architecture Guide

PolyMerge includes a pure-tensor `MoEAssembler` that constructs Mixture-of-Experts (MoE) models from individual dense models without requiring complex deep learning training harnesses.

---

## 1. Supported Target Architectures

### `mixtral` (Mistral / Mixtral-8x7B)
- **Structure:** 8 independent routed experts, Top-2 gating.
- **Tensors Created:**
  - `model.layers.{i}.block_sparse_moe.gate.weight`: Shape `[num_experts, hidden_size]`.
  - `model.layers.{i}.block_sparse_moe.experts.{e}.{w1,w2,w3}.weight`.

### `qwen` (Qwen2-MoE / Qwen2.5-MoE / Qwen3.5-MoE)
- **Structure:** Up to 64–512 experts with Top-8 routing and optional `shared_expert`.
- **Tensors Created:**
  - `model.layers.{i}.mlp.gate.weight`: Shape `[num_experts, hidden_size]`.
  - `model.layers.{i}.mlp.shared_expert.{gate_proj,up_proj,down_proj}.weight`.
  - `model.layers.{i}.mlp.experts.{e}.{gate_proj,up_proj,down_proj}.weight`.

### `deepseek` (DeepSeek-V2 / DeepSeek-V3)
- **Structure:** Fine-grained routed experts with Top-6/Top-8 gating and unconditional shared experts.
- **Tensors Created:**
  - `model.layers.{i}.mlp.gate.weight`.
  - `model.layers.{i}.mlp.shared_expert.*`.
  - `model.layers.{i}.mlp.experts.{e}.*`.

### `gemma` (Gemma 4 MoE 26B-A4B)
- **Structure:** 128 experts with 3-tensor routing subsystem.
- **Tensors Created:**
  - `model.layers.{i}.mlp.router.proj.weight`: Shape `[num_experts, hidden_size]`.
  - `model.layers.{i}.mlp.router.scale`: Input normalization vector `[hidden_size]`.
  - `model.layers.{i}.mlp.router.per_expert_scale`: Per-expert scale factor `[num_experts]`.
  - `model.layers.{i}.mlp.experts.{e}.{gate_proj,up_proj,down_proj}.weight`.

---

## 2. Gate Initialization Modes

- **`uniform_random` (Recommended):**
  Initializes router weights from $\mathcal{U}(-\text{scale}, \text{scale})$ where $\text{scale} = \sqrt{1 / d_{\text{hidden}}}$. Guarantees unbiased exploration and maximum router entropy on start.
- **`random` (Normal):**
  Initializes router weights from $\mathcal{N}(0, 0.02^2)$.

---

## 3. Configuration & CLI Usage

### YAML Config (`moe_config.yaml`)
```yaml
models:
  - deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
  - Qwen/Qwen2.5-Coder-7B-Instruct
  - meta-llama/Llama-3.1-8B-Instruct
base_model: Qwen/Qwen2.5-7B
shared_expert: Qwen/Qwen2.5-7B
target_format: qwen
gate_mode: uniform_random
chunk_size_bytes: 2000000000
```

### CLI Command
```bash
polymerge moe_config.yaml -o ./assembled_moe --moe
```
