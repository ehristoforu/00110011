# PolyMerge Documentation

## Overview
PolyMerge provides unified architecture-agnostic model merging across PyTorch Safetensors and GGUF quantized formats.

### Core Components
- `polymerge.filter.TensorFilter`: Regex-based layer filtering. Excludes vision encoders (`visual.*`, `vision_tower.*`, `mm_projector.*`) and allows targeted layer freezing via `skip_patterns`.
- `polymerge.methods`: Pure math implementations of state-of-the-art model merging methods.
- `polymerge.io`: Streamed chunked I/O handlers for Safetensors (`SafetensorsReader`, `SafetensorsWriter`) and GGUF files (`GGUFReaderWrapper`, `GGUFWriterWrapper`).
- `polymerge.engine.MergeEngine`: Direct and staged pipeline coordinator with RAM/VRAM optimization.

### Supported Merge Methods
| Method | Description | Base Model Required |
|---|---|---|
| `linear` | Weighted linear interpolation | No |
| `slerp` | Spherical linear interpolation between two models | No |
| `task_arithmetic` | Task vector addition ($W_b + \lambda \sum \tau_i$) | Yes |
| `ties` / `dare_ties` | DARE Bernoulli drop + TIES sign consensus & election | Yes |
| `della` | Magnitude-proportional drop (MAGPRUNE) + sign consensus | Yes |
| `omnimerge_v2` | OBIM top-$k$ + DAREx $1/q$ rescaling + EMR max-abs election | Yes |
| `model_stock` | Geometric weight averaging using pairwise angle consensus | Yes |
| `sce` | Softmax Certainty Election | Yes |
| `passthrough` | Passes through layers without alteration | No |

### Configuration Reference (YAML)
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
