import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from polymerge.io.safetensors_io import SafetensorsReader, SafetensorsWriter, resolve_model_path
from polymerge.utils import human_bytes, free_memory
from polymerge.ui.console import console, create_merge_progress

class LoRATool:
    @staticmethod
    def extract_lora(
        base_model: str,
        finetuned_model: str,
        output_dir: str,
        rank: int = 16,
        alpha: float = 32.0,
        target_modules: Optional[List[str]] = None,
    ):
        import torch
        from safetensors.torch import save_file

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        base_reader = SafetensorsReader(resolve_model_path(base_model))
        tuned_reader = SafetensorsReader(resolve_model_path(finetuned_model))

        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        console.print(f"[bold cyan]Extracting LoRA (rank={rank}, alpha={alpha}) from delta...[/bold cyan]")
        lora_weights: Dict[str, torch.Tensor] = {}

        common_keys = [k for k in base_reader.weight_map if k in tuned_reader.weight_map]
        total_extracted = 0

        with create_merge_progress() as progress:
            task = progress.add_task("[cyan]SVD Decomposition...", total=len(common_keys))

            for key in common_keys:
                progress.update(task, advance=1, description=f"[cyan]{key[:35]}")
                if not any(f".{m}.weight" in key for m in target_modules):
                    continue

                w_base = base_reader.get_tensor(key)
                w_tuned = tuned_reader.get_tensor(key)

                if w_base is None or w_tuned is None or not w_base.is_floating_point():
                    continue

                delta = (w_tuned.float() - w_base.float())
                u, s, vh = torch.linalg.svd(delta, full_matrices=False)

                r = min(rank, s.shape[0])
                sqrt_s = torch.diag(torch.sqrt(s[:r]))

                scale_factor = alpha / r
                w_a = (sqrt_s @ vh[:r])
                w_b = (u[:, :r] @ sqrt_s) / scale_factor

                mod_prefix = key.replace(".weight", "")
                lora_a_name = f"base_model.model.{mod_prefix}.lora_A.weight"
                lora_b_name = f"base_model.model.{mod_prefix}.lora_B.weight"

                lora_weights[lora_a_name] = w_a.to(w_base.dtype).contiguous()
                lora_weights[lora_b_name] = w_b.to(w_base.dtype).contiguous()
                total_extracted += 1
                free_memory()

        save_file(lora_weights, str(out_path / "adapter_model.safetensors"))

        adapter_config = {
            "base_model_name_or_path": str(base_model),
            "r": rank,
            "lora_alpha": alpha,
            "target_modules": target_modules,
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
        }
        with open(out_path / "adapter_config.json", "w", encoding="utf-8") as f:
            json.dump(adapter_config, f, indent=2)

        console.print(f"[bold green]✓ LoRA adapter extracted ({total_extracted} modules) saved to {out_path}[/bold green]")

    @staticmethod
    def merge_lora(
        base_model: str,
        adapter_path: str,
        output_dir: str,
        chunk_size_bytes: int = 2_000_000_000,
    ):
        import torch
        from safetensors import safe_open

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        base_reader = SafetensorsReader(resolve_model_path(base_model))
        writer = SafetensorsWriter(out_path, max_shard_bytes=chunk_size_bytes)

        adapter_dir = resolve_model_path(adapter_path)
        with open(adapter_dir / "adapter_config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)

        r = cfg.get("r", 16)
        alpha = cfg.get("lora_alpha", 32.0)
        scale = alpha / r

        adapter_safetensors = adapter_dir / "adapter_model.safetensors"
        adapter_weights = {}
        with safe_open(str(adapter_safetensors), framework="pt", device="cpu") as f:
            for k in f.keys():
                adapter_weights[k] = f.get_tensor(k)

        console.print(f"[bold cyan]Merging LoRA adapter (scale={scale:.2f}) into base model...[/bold cyan]")
        total_merged = 0
        total_bytes = 0

        with create_merge_progress() as progress:
            task = progress.add_task("[cyan]Merging LoRA...", total=len(base_reader.weight_map))

            for key in base_reader.weight_map.keys():
                progress.update(task, advance=1, description=f"[cyan]{key[:35]}")
                t_base = base_reader.get_tensor(key)
                if t_base is None:
                    continue

                mod_prefix = key.replace(".weight", "")
                lora_a_name = f"base_model.model.{mod_prefix}.lora_A.weight"
                lora_b_name = f"base_model.model.{mod_prefix}.lora_B.weight"

                if lora_a_name in adapter_weights and lora_b_name in adapter_weights:
                    wa = adapter_weights[lora_a_name].float()
                    wb = adapter_weights[lora_b_name].float()
                    delta = (wb @ wa) * scale
                    merged = (t_base.float() + delta).to(t_base.dtype)
                    writer.add_tensor(key, merged)
                    total_bytes += merged.numel() * merged.element_size()
                    total_merged += 1
                else:
                    writer.add_tensor(key, t_base)
                    total_bytes += t_base.numel() * t_base.element_size()

                free_memory()

        writer.finalize(total_size=total_bytes)

        for pat in ["tokenizer*", "*.json", "*.txt", "*.model"]:
            for f in base_reader.model_dir.glob(pat):
                if not f.name.endswith(".safetensors.index.json") and not f.name.endswith(".safetensors"):
                    dest = out_path / f.name
                    if not dest.exists():
                        import shutil
                        shutil.copy2(f, dest)

        console.print(f"[bold green]✓ LoRA Merged successfully! ({total_merged} layers modified)[/bold green]")

    @staticmethod
    def merge_adapters(
        adapters: List[str],
        weights: List[float],
        output_dir: str,
        method: str = "linear",
        density: float = 1.0,
    ):
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        adapter_dirs = [resolve_model_path(a) for a in adapters]
        adapter_dicts: List[Dict[str, torch.Tensor]] = []
        configs = []

        for a_dir in adapter_dirs:
            with open(a_dir / "adapter_config.json", "r", encoding="utf-8") as f:
                configs.append(json.load(f))
            d = {}
            with safe_open(str(a_dir / "adapter_model.safetensors"), framework="pt", device="cpu") as f:
                for k in f.keys():
                    d[k] = f.get_tensor(k)
            adapter_dicts.append(d)

        all_keys = list(adapter_dicts[0].keys())
        merged_weights: Dict[str, torch.Tensor] = {}

        w_sum = sum(weights)
        norm_weights = [w / w_sum for w in weights] if w_sum > 0 else weights

        console.print(f"[bold cyan]Merging {len(adapters)} LoRA adapters (method: {method})...[/bold cyan]")

        with create_merge_progress() as progress:
            task = progress.add_task("[cyan]Blending adapters...", total=len(all_keys))

            for key in all_keys:
                progress.update(task, advance=1, description=f"[cyan]{key[:35]}")
                tensors = [ad[key].float() for ad in adapter_dicts if key in ad]

                if len(tensors) != len(adapters):
                    merged_weights[key] = tensors[0].contiguous()
                    continue

                if method == "dare_ties" or method == "ties":
                    deltas = tensors
                    stacked = torch.stack(deltas, dim=0)
                    if density < 1.0:
                        mask = torch.rand_like(stacked) < density
                        stacked = torch.where(mask, stacked / density, torch.zeros_like(stacked))
                    summed = stacked.sum(dim=0)
                    consensus_sign = torch.sign(summed)
                    sign_match = torch.sign(stacked) == consensus_sign.unsqueeze(0)
                    kept = torch.where(sign_match, stacked, torch.zeros_like(stacked))
                    counts = sign_match.sum(dim=0).clamp(min=1)
                    res = kept.sum(dim=0) / counts
                else:
                    res = torch.zeros_like(tensors[0])
                    for t, w in zip(tensors, norm_weights):
                        res.add_(t, alpha=float(w))

                merged_weights[key] = res.to(tensors[0].dtype).contiguous()
                free_memory()

        save_file(merged_weights, str(out_path / "adapter_model.safetensors"))

        # Save config based on primary adapter
        primary_cfg = dict(configs[0])
        primary_cfg["merged_adapters"] = [str(a) for a in adapters]
        primary_cfg["merge_method"] = method
        with open(out_path / "adapter_config.json", "w", encoding="utf-8") as f:
            json.dump(primary_cfg, f, indent=2)

        console.print(f"[bold green]✓ Combined LoRA adapter saved to {out_path}[/bold green]")
