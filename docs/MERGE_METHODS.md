# Core Merging & Merge Methods

PolyMerge performs architecture-agnostic merging directly on PyTorch `safetensors` and quantized `gguf` files. It streams weights tensor-by-tensor, avoiding the memory overhead of loading full Hugging Face model classes.

---

## 1. Supported Merge Methods

### `linear` (Weighted Linear Interpolation)
Computes a weighted average of model weights:
$$W_{\text{merged}} = \sum_{i=1}^K w_i W_i$$
- **When to use:** Blending models with identical baselines or checkpoints from the same fine-tuning run.
- **Parameters:**
  - `normalize` (bool, default `true`): Normalizes weights such that $\sum w_i = 1.0$.

### `slerp` (Spherical Linear Interpolation)
Interpolates two models along the spherical hypersphere:
$$W_{\text{merged}} = \frac{\sin((1-t)\theta)}{\sin\theta} W_1 + \frac{\sin(t\theta)}{\sin\theta} W_2$$
- **When to use:** Merging exactly 2 models to preserve directional geometric features.
- **Parameters:**
  - `weight` (float, default `0.5`): Interpolation factor $t \in [0.0, 1.0]$.

### `task_arithmetic`
Adds task vectors ($\tau_i = W_i - W_{\text{base}}$) to the base model:
$$W_{\text{merged}} = W_{\text{base}} + \lambda \sum_{i=1}^K w_i (W_i - W_{\text{base}})$$
- **When to use:** Combining distinct capabilities (e.g. Code + Math + Chat) into a single base model.
- **Parameters:**
  - `scaling_factor` (float, default `1.0`): Global multiplier $\lambda$.

### `ties` / `dare_ties`
Trim, Elect Sign, and Merge with DARE Bernoulli drop:
1. **DARE Drop:** Drop random parameter deltas with keep probability $d$, rescale by $1/d$.
2. **TIES Sign Consensus:** Compute consensus sign $\text{sign}(\sum \delta_i)$ and zero out conflicting signs.
3. **Disjoint Merge:** Average only surviving, non-conflicting deltas.
- **Parameters:**
  - `density` (float, default `0.5`): Keep probability $d \in (0.0, 1.0]$.
  - `weight` (float, default `1.0`): Source weight.

### `della` (MAGPRUNE Drop)
Replaces uniform Bernoulli drop with magnitude-proportional rank drop (MAGPRUNE):
- Parameters with larger absolute deltas have higher survival probability:
  $$p = \min\left(1.0, \frac{|\delta|}{\text{threshold}} \cdot d\right) \cdot (1 - \epsilon) + \epsilon$$
- **Parameters:**
  - `density` (float, default `0.7`): Keep density.
  - `epsilon` (float, default `0.15`): Minimum probability floor.

### `omnimerge_v2`
The state-of-the-art method from `omnimergekit`:
1. **OBIM-lite:** Deterministic top-$k$ masking by $|\delta|$ (preserves most impactful weights).
2. **DAREx:** Rescaling by $1/q$ rather than $1/d$, preventing variance explosion.
3. **EMR Election:** Elects consensus sign from weighted sum, amplitude from maximum absolute delta.
- **Parameters:**
  - `darex_q` (float, default `0.75`): Scaling denominator.
  - `features` (list of strings, default `["obim", "darex", "emr"]`).

### `model_stock`
Geometric angle consensus:
Calculates pairwise cosine similarities between source deltas and computes optimal scalar $\gamma$:
$$\gamma = \frac{K \bar{c}}{1 + (K-1)\bar{c}}$$
- **When to use:** Merging 3+ fine-tunes of the same base model.

### `sce` (Softmax Certainty Election)
Applies elementwise softmax across source delta magnitudes:
$$W_{\text{merged}} = W_{\text{base}} + \sum_{i=1}^K \text{softmax}(|\delta|)_i \cdot \delta_i$$

### `karcher` (Karcher Mean / Riemannian Barycenter)
Computes the true geometric mean on the unit hypersphere manifold via Riemannian gradient descent. Highly effective when combining 3 or more models without euclidean distortion.

### `nuslerp` (Multi-Dimensional NuSLERP)
N-dimensional spherical linear interpolation. Properly handles collinear branches to avoid gradient/norm singularities.

### `nearswap`
Locally swaps parameters based on weight delta proximity, creating smooth continuous blending between homologous architectures.

---

## 2. Full YAML Configuration Example

```yaml
merge_method: omnimerge_v2
base_model: Qwen/Qwen2.5-7B
models:
  - model: Qwen/Qwen2.5-Coder-7B-Instruct
    weight: 0.5
    density: 0.6
  - model: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    weight: 0.5
    density: 0.6
parameters:
  darex_q: 0.75
  features: ["obim", "darex", "emr"]
dtype: bfloat16
ignore_vision: true
skip_patterns:
  - "mlp.gate_proj"
  - "mlp.up_proj"
  - "mlp.down_proj"
chunk_size_bytes: 2000000000
device: auto
```

---

## 3. GGUF Direct Merging
PolyMerge can merge pre-quantized models (e.g. Q4_K_M or Q8_0) directly into a final GGUF file without first dequantizing the entire model to FP16 disk shards:
```python
from polymerge.engine_gguf import GGUFMergeEngine

engine = GGUFMergeEngine(
    base_gguf="model_base.Q4_K_M.gguf",
    source_ggufs=["model_a.Q4_K_M.gguf", "model_b.Q4_K_M.gguf"],
    weights=[0.5, 0.5],
    densities=[0.7, 0.7],
    merge_method="linear",
)
engine.merge_to_gguf("merged_model.Q4_K_M.gguf")
```
