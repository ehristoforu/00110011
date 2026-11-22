# Layer Slicing & Frankenmerging Guide

Layer slicing (often referred to as **Frankenmerging** or depth-extension) allows stacking and interpolating layer ranges from different models to increase total depth and capacity (e.g. converting a 32-layer 7B model into a 44-layer or 48-layer 10B/11B model).

---

## 1. How Layer Slicing Works
PolyMerge's `LayerSlicer` walks through the requested slice ranges, re-indexes the transformer layers consecutively starting from index 0, and updates the `num_hidden_layers` in the output `config.json`.

- Non-layer parameters (embeddings, normalization, final `lm_head`) are preserved from the first base model.
- Intermediate layer weights (`model.layers.{i}.*`) are remapped sequentially.

---

## 2. YAML Configuration Specification

```yaml
slices:
  - sources:
      - model: meta-llama/Llama-3.2-1B
        layer_range: [0, 12]       # Layers 0 to 11
  - sources:
      - model: meta-llama/Llama-3.2-1B-Instruct
        layer_range: [8, 16]       # Layers 8 to 15 (overlapping depth)
merge_method: passthrough
dtype: bfloat16
```

### CLI Command
```bash
polymerge slice_config.yaml -o ./frankenmerged_model --slices
```
