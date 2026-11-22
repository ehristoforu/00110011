# Model Pruning & Compression Guide

PolyMerge implements zero-backprop pruning algorithms inspired by `omnimergekit` (REAP, T17, and T18 frameworks). Models are compressed without requiring fine-tuning datasets or expensive gradient passes.

---

## 1. Dense Layer Pruning (`--prune-dense`)

### Algorithm & Mathematical Foundations
Dense transformers contain deep layers whose outputs are near-identity transformations relative to adjacent blocks. PolyMerge detects and prunes these redundant layers:

1. **Cosine Block Similarity Analysis:**
   For every layer $l \in [1, L-1]$, we calculate the cosine similarity between adjacent projection layers:
   $$S(l, l+1) = \frac{\langle W_l, W_{l+1} \rangle}{\|W_l\|_2 \|W_{l+1}\|_2}$$
   Layers with the highest cosine similarity represent redundant features and are scheduled for pruning.
2. **Boundary Protection:**
   The first 2 layers and final 2 layers are strictly protected to prevent catastrophic loss of token embeddings and logit heads.
3. **Local Healing (Residual Compensation):**
   When layer $l$ is removed, the remaining subsequent layer's down-projection / output-projection is rescaled to heal the residual stream:
   $$W_{\text{healed}} = W \cdot \left(1 + \frac{1}{L_{\text{remaining}}}\right)$$

### CLI Usage
```bash
# Auto-detect and prune 25% most redundant layers
polymerge --prune-dense --base /path/to/llama-3-8b -o ./llama-pruned --drop-fraction 0.25

# Explicitly prune specific layer indices
polymerge --prune-dense --base /path/to/llama-3-8b -o ./llama-pruned --drop-layers 12 13 14 15
```

---

## 2. MoE Expert Pruning (`--prune-moe`)

### Problem: Router Instability & Rumination
Dropping experts without router recalibration produces severe pathological behaviors:
- **Rumination:** The model loops in thinking tokens, dumping reasoning until budget exhaustion.
- **Silent-Empty:** The router selects dropped slots, resulting in empty string generation.

### Solution: T18 Soft-Transfer Router Recovery
1. **L2 Magnitude Profiling:**
   Experts are ranked by their router projection weight norms:
   $$\text{Score}(e) = \|W_{\text{router}}[e]\|_2$$
   The top-$K$ experts are kept, and the bottom experts $\mathcal{D}$ are dropped.
2. **Soft-Transfer Cosine Injection:**
   For each dropped expert $d \in \mathcal{D}$, we compute its cosine similarity with all surviving experts $\mathcal{S}$. We inject a fraction $\alpha$ of $d$'s router row into the top-$k$ nearest surviving experts:
   $$W_{\text{router}}[s] \leftarrow W_{\text{router}}[s] + \alpha \cdot \frac{\cos(W[s], W[d])}{\sum_{k} \cos(W[k], W[d])} \cdot W_{\text{router}}[d]$$
   - Default transfer factor: $\alpha = 0.3$ (empirically prevents 100% of rumination loops).

### CLI Usage
```bash
polymerge --prune-moe --base google/gemma-4-26B-A4B-it -o ./gemma-64e --target-experts 64 --alpha 0.3
```
