from typing import List, Optional, Dict, Any

try:
    import torch
    from polymerge.methods.base import BaseMergeMethod
except ImportError:
    class _MockTorch:
        Tensor = object
    torch = _MockTorch()
    BaseMergeMethod = object

class LinearMerge(BaseMergeMethod):
    def merge(
        self,
        base_tensor: Optional[torch.Tensor],
        source_tensors: List[torch.Tensor],
        weights: List[float],
        densities: List[float],
        parameters: Dict[str, Any],
    ) -> torch.Tensor:
        orig_dtype = source_tensors[0].dtype
        w_sum = sum(weights)
        norm_weights = [w / w_sum for w in weights] if parameters.get("normalize", True) and w_sum > 0 else weights

        res = torch.zeros_like(source_tensors[0], dtype=torch.float32)
        for t, w in zip(source_tensors, norm_weights):
            res.add_(t.to(torch.float32), alpha=float(w))
        return res.to(orig_dtype)

class SlerpMerge(BaseMergeMethod):
    def merge(
        self,
        base_tensor: Optional[torch.Tensor],
        source_tensors: List[torch.Tensor],
        weights: List[float],
        densities: List[float],
        parameters: Dict[str, Any],
    ) -> torch.Tensor:
        orig_dtype = source_tensors[0].dtype
        t0 = source_tensors[0].to(torch.float32)
        t1 = source_tensors[1].to(torch.float32) if len(source_tensors) > 1 else t0
        t = float(weights[1] if len(weights) > 1 else 0.5)

        t0_norm = t0 / torch.norm(t0).clamp(min=1e-8)
        t1_norm = t1 / torch.norm(t1).clamp(min=1e-8)
        dot = (t0_norm * t1_norm).sum().clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        omega = torch.acos(dot)
        sin_omega = torch.sin(omega)
        if sin_omega.abs() < 1e-6:
            res = (1.0 - t) * t0 + t * t1
        else:
            scale0 = torch.sin((1.0 - t) * omega) / sin_omega
            scale1 = torch.sin(t * omega) / sin_omega
            res = scale0 * t0 + scale1 * t1
        return res.to(orig_dtype)

class TaskArithmeticMerge(BaseMergeMethod):
    def merge(
        self,
        base_tensor: Optional[torch.Tensor],
        source_tensors: List[torch.Tensor],
        weights: List[float],
        densities: List[float],
        parameters: Dict[str, Any],
    ) -> torch.Tensor:
        if base_tensor is None:
            raise ValueError("Task Arithmetic requires a base_model")
        orig_dtype = base_tensor.dtype
        base_f32 = base_tensor.to(torch.float32)
        merged_delta = torch.zeros_like(base_f32)

        for s, w in zip(source_tensors, weights):
            delta = s.to(torch.float32) - base_f32
            merged_delta.add_(delta, alpha=float(w))

        scale = float(parameters.get("scaling_factor", 1.0))
        res = base_f32 + merged_delta * scale
        return res.to(orig_dtype)

class DareTiesMerge(BaseMergeMethod):
    def merge(
        self,
        base_tensor: Optional[torch.Tensor],
        source_tensors: List[torch.Tensor],
        weights: List[float],
        densities: List[float],
        parameters: Dict[str, Any],
    ) -> torch.Tensor:
        if base_tensor is None:
            raise ValueError("DARE-TIES requires a base_model")
        orig_dtype = base_tensor.dtype
        base_f32 = base_tensor.to(torch.float32)
        masked_deltas = []

        for s, w, d in zip(source_tensors, weights, densities):
            delta = s.to(torch.float32) - base_f32
            if d < 1.0:
                mask = torch.rand_like(delta) < d
                delta = torch.where(mask, delta / d, torch.zeros_like(delta))
            masked_deltas.append(delta * float(w))

        stack = torch.stack(masked_deltas, dim=0)
        summed = stack.sum(dim=0)
        consensus_sign = torch.sign(summed)

        sign_match = torch.sign(stack) == consensus_sign.unsqueeze(0)
        kept = torch.where(sign_match, stack, torch.zeros_like(stack))

        non_zero_counts = sign_match.sum(dim=0).clamp(min=1)
        merged_delta = kept.sum(dim=0) / non_zero_counts
        res = base_f32 + merged_delta
        return res.to(orig_dtype)

class DellaMerge(BaseMergeMethod):
    def merge(
        self,
        base_tensor: Optional[torch.Tensor],
        source_tensors: List[torch.Tensor],
        weights: List[float],
        densities: List[float],
        parameters: Dict[str, Any],
    ) -> torch.Tensor:
        if base_tensor is None:
            raise ValueError("DELLA requires a base_model")
        orig_dtype = base_tensor.dtype
        base_f32 = base_tensor.to(torch.float32)
        epsilon = float(parameters.get("epsilon", 0.15))
        masked_deltas = []

        for s, w, d in zip(source_tensors, weights, densities):
            delta = s.to(torch.float32) - base_f32
            if d < 1.0:
                abs_d = delta.abs()
                k = max(1, int(d * delta.numel()))
                top_vals, _ = torch.topk(abs_d.reshape(-1), k, largest=True)
                threshold = top_vals[-1]
                prob = torch.clamp((abs_d / threshold.clamp(min=1e-8)) * d, max=1.0)
                mask = torch.rand_like(delta) < (prob * (1.0 - epsilon) + epsilon)
                delta = torch.where(mask, delta / (prob.clamp(min=1e-4)), torch.zeros_like(delta))
            masked_deltas.append(delta * float(w))

        stack = torch.stack(masked_deltas, dim=0)
        summed = stack.sum(dim=0)
        consensus_sign = torch.sign(summed)
        sign_match = torch.sign(stack) == consensus_sign.unsqueeze(0)
        kept = torch.where(sign_match, stack, torch.zeros_like(stack))
        counts = sign_match.sum(dim=0).clamp(min=1)
        merged_delta = kept.sum(dim=0) / counts
        return (base_f32 + merged_delta).to(orig_dtype)

class OmnimergeV2Merge(BaseMergeMethod):
    def merge(
        self,
        base_tensor: Optional[torch.Tensor],
        source_tensors: List[torch.Tensor],
        weights: List[float],
        densities: List[float],
        parameters: Dict[str, Any],
    ) -> torch.Tensor:
        if base_tensor is None:
            raise ValueError("Omnimerge v2 requires a base_model")
        orig_dtype = base_tensor.dtype
        base_f32 = base_tensor.to(torch.float32)
        darex_q = float(parameters.get("darex_q", 0.75))
        features = parameters.get("features", ["obim", "darex", "emr"])
        masked = []

        for s, w, d in zip(source_tensors, weights, densities):
            delta = s.to(torch.float32) - base_f32
            numel = delta.numel()
            k = max(1, int(d * numel))
            if d < 1.0:
                if "obim" in features:
                    flat = delta.abs().reshape(-1)
                    _, top_idx = torch.topk(flat, k, largest=True, sorted=False)
                    mask = torch.zeros(numel, dtype=torch.bool, device=delta.device)
                    mask[top_idx] = True
                    mask = mask.reshape_as(delta)
                else:
                    mask = torch.rand_like(delta) < d

                if "darex" in features:
                    delta = torch.where(mask, delta / darex_q, torch.zeros_like(delta))
                else:
                    delta = torch.where(mask, delta / d, torch.zeros_like(delta))
            masked.append(delta * float(w))

        stack = torch.stack(masked, dim=0)
        if "emr" in features:
            weighted_sum = stack.sum(dim=0)
            consensus_sign = torch.sign(weighted_sum)
            max_abs, _ = stack.abs().max(dim=0)
            merged_delta = consensus_sign * max_abs
            any_agree = (torch.sign(stack) == consensus_sign.unsqueeze(0)).any(dim=0)
            merged_delta = torch.where(any_agree, merged_delta, torch.zeros_like(merged_delta))
        else:
            summed = stack.sum(dim=0)
            consensus_sign = torch.sign(summed)
            sign_match = torch.sign(stack) == consensus_sign.unsqueeze(0)
            kept = torch.where(sign_match, stack, torch.zeros_like(stack))
            counts = sign_match.sum(dim=0).clamp(min=1)
            merged_delta = kept.sum(dim=0) / counts

        return (base_f32 + merged_delta).to(orig_dtype)

class ModelStockMerge(BaseMergeMethod):
    def merge(
        self,
        base_tensor: Optional[torch.Tensor],
        source_tensors: List[torch.Tensor],
        weights: List[float],
        densities: List[float],
        parameters: Dict[str, Any],
    ) -> torch.Tensor:
        if base_tensor is None or len(source_tensors) < 2:
            return LinearMerge().merge(base_tensor, source_tensors, weights, densities, parameters)
        orig_dtype = base_tensor.dtype
        base_f32 = base_tensor.to(torch.float32)
        deltas = [(s.to(torch.float32) - base_f32).flatten() for s in source_tensors]

        cos_sims = []
        for i in range(len(deltas)):
            for j in range(i + 1, len(deltas)):
                d0, d1 = deltas[i], deltas[j]
                n0 = torch.norm(d0).clamp(min=1e-8)
                n1 = torch.norm(d1).clamp(min=1e-8)
                cos_sims.append((d0 @ d1) / (n0 * n1))
        avg_cos = torch.stack(cos_sims).mean().clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        K = len(source_tensors)
        gamma = (K * avg_cos) / (1.0 + (K - 1) * avg_cos) if (1.0 + (K - 1) * avg_cos).abs() > 1e-6 else 1.0

        mean_delta = torch.stack([s.to(torch.float32) - base_f32 for s in source_tensors], dim=0).mean(dim=0)
        res = base_f32 + gamma * mean_delta
        return res.to(orig_dtype)

class SCEMerge(BaseMergeMethod):
    def merge(
        self,
        base_tensor: Optional[torch.Tensor],
        source_tensors: List[torch.Tensor],
        weights: List[float],
        densities: List[float],
        parameters: Dict[str, Any],
    ) -> torch.Tensor:
        if base_tensor is None:
            return LinearMerge().merge(base_tensor, source_tensors, weights, densities, parameters)
        orig_dtype = base_tensor.dtype
        base_f32 = base_tensor.to(torch.float32)
        deltas = [s.to(torch.float32) - base_f32 for s in source_tensors]
        stack = torch.stack(deltas, dim=0)

        mag = stack.abs()
        weights_sce = torch.softmax(mag, dim=0)
        merged_delta = (stack * weights_sce).sum(dim=0)
        return (base_f32 + merged_delta).to(orig_dtype)

class PassthroughMerge(BaseMergeMethod):
    def merge(
        self,
        base_tensor: Optional[torch.Tensor],
        source_tensors: List[torch.Tensor],
        weights: List[float],
        densities: List[float],
        parameters: Dict[str, Any],
    ) -> torch.Tensor:
        target = source_tensors[0] if source_tensors else base_tensor
        return target.clone()

try:
    from polymerge.methods.advanced import KarcherMerge, NuSlerpMerge, NearSwapMerge
except ImportError:
    KarcherMerge = object
    NuSlerpMerge = object
    NearSwapMerge = object

METHODS_REGISTRY: Dict[str, BaseMergeMethod] = {
    "linear": LinearMerge(),
    "slerp": SlerpMerge(),
    "nuslerp": NuSlerpMerge(),
    "karcher": KarcherMerge(),
    "nearswap": NearSwapMerge(),
    "task_arithmetic": TaskArithmeticMerge(),
    "dare_ties": DareTiesMerge(),
    "ties": DareTiesMerge(),
    "della": DellaMerge(),
    "omnimerge_v2": OmnimergeV2Merge(),
    "model_stock": ModelStockMerge(),
    "sce": SCEMerge(),
    "passthrough": PassthroughMerge(),
}

def get_merge_method(name: str) -> BaseMergeMethod:
    method = METHODS_REGISTRY.get(name.lower())
    if not method:
        raise ValueError(f"Unknown merge method: {name}. Available: {list(METHODS_REGISTRY.keys())}")
    return method
