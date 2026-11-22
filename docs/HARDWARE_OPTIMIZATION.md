# Hardware, Memory & Staged Optimization

PolyMerge is designed to execute on resource-constrained domestic PCs as well as multi-accelerator clusters.

---

## 1. Multi-Device Execution (`--device`)

PolyMerge auto-detects and routes tensor computation to the highest-performing available accelerator:
- `auto`: Prioritizes TPU (`torch_xla`), then CUDA GPU (`torch.cuda`), falling back to CPU.
- `cpu`: Optimized for systems with high system RAM.
- `cuda`: Offloads tensor mathematics to NVIDIA GPUs.
- `tpu`: Offloads matrix operations to Google Cloud TPUs via `torch_xla`.

### Example
```bash
polymerge config.yaml -o ./merged_model --device cuda
```

---

## 2. Low-RAM Staged Merging (`staged: true`)

On domestic PCs or low-RAM environments (e.g., 16 GB RAM merging two 70B models), attempting to hold all deltas simultaneously causes Out-Of-Memory (OOM) crashes. 

PolyMerge supports staged multi-pass pipelines:
```yaml
staged: true
stages:
  - name: "Stage 1: Blending Instruct Checkpoints"
    merge_method: linear
    models:
      - model: instruct_a
        weight: 0.5
      - model: instruct_b
        weight: 0.5
    dtype: bfloat16

  - name: "Stage 2: DARE-TIES with Foundation Base"
    merge_method: dare_ties
    base_model: foundation_base
    models:
      - model: PREVIOUS_STAGE
        weight: 0.7
        density: 0.6
      - model: domain_specialist
        weight: 0.3
        density: 0.6
    dtype: bfloat16
```
Each stage writes its intermediate checkpoint and drops page caches via `drop_caches` before launching the subsequent stage.

---

## 3. Fast Hugging Face Transfer

PolyMerge integrates Rust-accelerated parallel downloaders:
- `hf-transfer`: Multi-threaded parallel file transfer (`HF_HUB_ENABLE_HF_TRANSFER=1`).
- `hf-xet`: Deduplication and chunk-based Xet repository streaming.
Both are automatically engaged when pulling remote models.
