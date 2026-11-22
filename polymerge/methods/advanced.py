from typing import List, Optional, Dict, Any

try:
    import torch
    from polymerge.methods.base import BaseMergeMethod
except ImportError:
    class _MockTorch:
        Tensor = object
    torch = _MockTorch()
    BaseMergeMethod = object

class KarcherMerge(BaseMergeMethod):
    def merge(
        self,
        base_tensor: Optional[torch.Tensor],
        source_tensors: List[torch.Tensor],
        weights: List[float],
        densities: List[float],
        parameters: Dict[str, Any],
    ) -> torch.Tensor:
        if len(source_tensors) == 1:
            return source_tensors[0]

        max_iter = int(parameters.get("max_iter", 10))
        tol = float(parameters.get("tol", 1e-5))
        orig_dtype = source_tensors[0].dtype

        w_sum = sum(weights)
        alphas = [w / w_sum for w in weights] if w_sum > 0 else [1.0 / len(source_tensors)] * len(source_tensors)

        dtype = torch.float64 if orig_dtype == torch.float64 else torch.float32
        norms = []
        units = []
        for tensor in source_tensors:
            norm = torch.linalg.vector_norm(tensor.to(dtype))
            norms.append(norm)
            if norm.item() == 0.0:
                units.append(torch.zeros_like(tensor, dtype=dtype))
            else:
                units.append((tensor / norm).to(dtype))

        mean = torch.zeros_like(units[0])
        for alpha, unit in zip(alphas, units):
            mean += alpha * unit
        mean_norm = torch.linalg.vector_norm(mean)
        if mean_norm.item() > tol:
            mean = mean / mean_norm

        for _ in range(max_iter):
            tangent = torch.zeros_like(mean)
            for alpha, unit in zip(alphas, units):
                dot = torch.clamp(torch.dot(mean.flatten(), unit.flatten()), -1.0, 1.0)
                theta = torch.arccos(dot)
                if theta.item() < tol:
                    continue
                tangent += alpha * (theta / torch.sin(theta)) * (unit - dot * mean)

            tangent_norm = torch.linalg.vector_norm(tangent)
            if tangent_norm.item() < tol:
                break
            mean = torch.cos(tangent_norm) * mean + torch.sin(tangent_norm) * (tangent / tangent_norm)
            unit_norm = torch.linalg.vector_norm(mean)
            if unit_norm.item() > tol:
                mean = mean / unit_norm

        scale = sum(alpha * norm for alpha, norm in zip(alphas, norms))
        return (scale * mean).to(orig_dtype)

class NuSlerpMerge(BaseMergeMethod):
    def merge(
        self,
        base_tensor: Optional[torch.Tensor],
        source_tensors: List[torch.Tensor],
        weights: List[float],
        densities: List[float],
        parameters: Dict[str, Any],
    ) -> torch.Tensor:
        orig_dtype = source_tensors[0].dtype
        w_sum = sum(weights[:2]) if len(weights) >= 2 else 1.0
        t = float(weights[1] / w_sum) if len(weights) >= 2 and w_sum > 0 else 0.5

        v0 = source_tensors[0].float()
        v1 = source_tensors[1].float() if len(source_tensors) > 1 else v0

        if base_tensor is not None:
            base_f = base_tensor.float()
            v0 = v0 - base_f
            v1 = v1 - base_f

        v0_flat = v0.flatten()
        v1_flat = v1.flatten()
        v0_u = v0_flat / torch.norm(v0_flat).clamp(min=1e-7)
        v1_u = v1_flat / torch.norm(v1_flat).clamp(min=1e-7)

        dot = torch.clamp(torch.dot(v0_u, v1_u), -1.0, 1.0)
        colinear = (1.0 - dot.square()) <= 1e-12

        if colinear.item():
            res = (1.0 - t) * v0 + t * v1
        else:
            theta = torch.arccos(dot)
            sin_theta = torch.sin(theta)
            res = (torch.sin((1.0 - t) * theta) * v0 + torch.sin(t * theta) * v1) / sin_theta

        if base_tensor is not None:
            res = res + base_f

        return res.to(orig_dtype)

class NearSwapMerge(BaseMergeMethod):
    def merge(
        self,
        base_tensor: Optional[torch.Tensor],
        source_tensors: List[torch.Tensor],
        weights: List[float],
        densities: List[float],
        parameters: Dict[str, Any],
    ) -> torch.Tensor:
        if base_tensor is None:
            return source_tensors[0]
        orig_dtype = base_tensor.dtype
        a = base_tensor.float()
        b = source_tensors[0].float()
        t = float(parameters.get("t", weights[0] if weights else 0.001))

        absdiff = torch.abs(a - b)
        w = (t / absdiff.clamp(min=1e-6)).clamp(min=0.0, max=1.0)
        res = w * b + (1.0 - w) * a
        return res.to(orig_dtype)
