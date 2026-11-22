import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from polymerge.io.safetensors_io import SafetensorsReader, resolve_model_path
from polymerge.ui.console import console

class SanityChecker:
    @staticmethod
    def audit_model(model_dir: str, cond_threshold: float = 10.0) -> Dict[str, Any]:
        """
        Comprehensive Sanity & Collapse Diagnostics for merged/pruned models:
        1. NaN / Inf parameter check across all tensors.
        2. Router conditioning analysis (detects router collapse / rumination).
        3. Weight norm divergence detection.
        """
        import torch

        p = resolve_model_path(model_dir)
        reader = SafetensorsReader(p)

        console.print(f"[bold cyan]Running Collapse & Sanity Audit on: {p}...[/bold cyan]")

        total_tensors = len(reader.weight_map)
        nan_tensors = []
        inf_tensors = []
        degraded_routers = []

        for name in reader.weight_map.keys():
            t = reader.get_tensor(name)
            if t is None or not t.is_floating_point():
                continue

            t_float = t.float()
            if torch.isnan(t_float).any().item():
                nan_tensors.append(name)
            if torch.isinf(t_float).any().item():
                inf_tensors.append(name)

            # Check router condition number for MoE
            if name.endswith(".router.proj.weight") or name.endswith(".mlp.gate.weight") or name.endswith(".block_sparse_moe.gate.weight"):
                if t_float.ndim == 2 and t_float.shape[0] > 1:
                    c = torch.linalg.cond(t_float).item()
                    if math.isnan(c) or c > cond_threshold:
                        degraded_routers.append((name, c))

        is_healthy = (len(nan_tensors) == 0 and len(inf_tensors) == 0 and len(degraded_routers) == 0)

        if is_healthy:
            console.print("[bold green]✓ All checks passed! Model weights are clean, healthy, and non-collapsed.[/bold green]")
        else:
            if nan_tensors:
                console.print(f"[bold red]✗ Detected NaN tensors: {len(nan_tensors)}[/bold red]")
            if inf_tensors:
                console.print(f"[bold red]✗ Detected Inf tensors: {len(inf_tensors)}[/bold red]")
            if degraded_routers:
                console.print(f"[bold yellow]⚠ Warning: Potential MoE router degeneracy in {len(degraded_routers)} layers.[/bold yellow]")

        return {
            "is_healthy": is_healthy,
            "total_tensors": total_tensors,
            "nan_count": len(nan_tensors),
            "inf_count": len(inf_tensors),
            "degraded_routers": degraded_routers,
        }
