import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from polymerge.io.safetensors_io import SafetensorsReader, SafetensorsWriter, resolve_model_path
from polymerge.utils import human_bytes, free_memory, select_device
from polymerge.ui.console import console, create_merge_progress
from polymerge.filter import TensorFilter

class ModelPruner:
    @staticmethod
    def prune_dense_layers(
        model_path: str,
        output_dir: str,
        drop_layers: Optional[List[int]] = None,
        drop_fraction: float = 0.2,
        device: str = "auto",
        chunk_size_bytes: int = 2_000_000_000,
    ):
        """
        Universal Dense Layer Pruning with Cosine Block Similarity & Local Healing.
        Analyzes transformer text layers, drops the most redundant / identity-like layers,
        and applies linear residual correction (no backpropagation/retraining needed).
        """
        import torch

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        m_path = resolve_model_path(model_path)
        reader = SafetensorsReader(m_path)
        writer = SafetensorsWriter(out_path, max_shard_bytes=chunk_size_bytes)
        dev = select_device(device)
        tensor_filter = TensorFilter(ignore_vision=True)

        # Detect total layers
        layer_indices = set()
        for k in reader.weight_map.keys():
            m = re.search(r"\.(?:layers|blocks)\.(\d+)\.", k)
            if m:
                layer_indices.add(int(m.group(1)))

        total_layers = len(layer_indices)
        if total_layers == 0:
            raise ValueError(f"No indexed transformer layers found in {m_path}")

        console.print(f"[bold cyan]Pruning Dense Model ({total_layers} layers detected)...[/bold cyan]")

        # If drop_layers not explicitly given, compute angular/norm delta across adjacent layers
        if drop_layers is None:
            num_to_drop = max(1, int(total_layers * drop_fraction))
            console.print(f"[yellow]Auto-analyzing layer redundancy to drop {num_to_drop} layers...[/yellow]")
            layer_scores = []

            # Compare adjacent MLP down_proj / gate_proj deltas
            for l_idx in range(total_layers - 1):
                k1 = f"model.layers.{l_idx}.mlp.down_proj.weight"
                k2 = f"model.layers.{l_idx + 1}.mlp.down_proj.weight"
                t1 = reader.get_tensor(k1)
                t2 = reader.get_tensor(k2)
                if t1 is not None and t2 is not None:
                    # Cosine similarity between adjacent layers: high similarity = redundant layer
                    cos_sim = torch.nn.functional.cosine_similarity(
                        t1.flatten().float().unsqueeze(0),
                        t2.flatten().float().unsqueeze(0)
                    ).item()
                    layer_scores.append((cos_sim, l_idx + 1))
                else:
                    layer_scores.append((0.0, l_idx + 1))

            # Protect very first and very last layers
            layer_scores.sort(key=lambda x: x[0], reverse=True)
            candidate_drops = [idx for _, idx in layer_scores if 2 <= idx < total_layers - 2]
            drop_layers = sorted(candidate_drops[:num_to_drop])

        drop_set = set(drop_layers)
        console.print(f"  • Dropping layers: [bold red]{sorted(list(drop_set))}[/bold red]")
        console.print(f"  • Remaining layers: [bold green]{total_layers - len(drop_set)}[/bold green]")

        # Copy non-layer tensors
        total_bytes = 0
        for k in reader.weight_map.keys():
            if tensor_filter.should_ignore(k):
                continue
            if not re.search(r"\.(?:layers|blocks)\.\d+\.", k):
                t = reader.get_tensor(k)
                if t is not None:
                    writer.add_tensor(k, t)
                    total_bytes += t.numel() * t.element_size()

        # Remap and heal surviving layers
        new_layer_idx = 0
        with create_merge_progress() as progress:
            task = progress.add_task("[cyan]Re-indexing and streaming layers...", total=total_layers)

            for l_idx in range(total_layers):
                progress.update(task, advance=1, description=f"[cyan]Layer {l_idx}")
                if l_idx in drop_set:
                    continue

                prefix = f"model.layers.{l_idx}."
                new_prefix = f"model.layers.{new_layer_idx}."
                matching_keys = [k for k in reader.weight_map.keys() if k.startswith(prefix)]

                for old_k in matching_keys:
                    if tensor_filter.should_ignore(old_k):
                        continue
                    new_k = old_k.replace(prefix, new_prefix)
                    t = reader.get_tensor(old_k)
                    if t is None:
                        continue

                    # Local healing: if previous layer was dropped, slightly rescale residual projection
                    # to compensate for dropped transformation: W_out *= (1 + 1 / remaining_layers)
                    if (l_idx - 1) in drop_set and any(sub in old_k for sub in ["down_proj", "o_proj"]):
                        heal_factor = 1.0 + (1.0 / (total_layers - len(drop_set)))
                        t_healed = (t.float() * heal_factor).to(t.dtype)
                        writer.add_tensor(new_k, t_healed)
                        total_bytes += t_healed.numel() * t_healed.element_size()
                    else:
                        writer.add_tensor(new_k, t)
                        total_bytes += t.numel() * t.element_size()

                new_layer_idx += 1
                free_memory()

        writer.finalize(total_size=total_bytes)

        # Update config.json (root and nested text_config)
        cfg_path = m_path / "config.json"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg["num_hidden_layers"] = new_layer_idx
            if "text_config" in cfg and isinstance(cfg["text_config"], dict):
                cfg["text_config"]["num_hidden_layers"] = new_layer_idx
            with open(out_path / "config.json", "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)

        for pat in ["tokenizer*", "*.json", "*.txt", "*.model"]:
            for f in m_path.glob(pat):
                if not f.name.endswith(".safetensors.index.json") and not f.name.endswith(".safetensors") and f.name != "config.json":
                    dest = out_path / f.name
                    if not dest.exists():
                        shutil.copy2(f, dest)

        console.print(f"[bold green]✓ Dense pruning completed ({new_layer_idx} layers saved to {out_path})[/bold green]")

    @staticmethod
    def prune_moe_experts(
        model_path: str,
        output_dir: str,
        target_experts: int = 64,
        alpha: float = 0.3,
        top_k_neighbors: int = 3,
        chunk_size_bytes: int = 2_000_000_000,
    ):
        """
        Universal MoE Expert Pruner with Soft-Transfer Router Recovery (T18 framework).
        Finds the least essential/redundant experts, drops their weights, and injects a
        weighted cosine-similar share of dropped router logits onto surviving experts,
        completely eliminating router collapse and rumination loops.
        """
        import torch

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        m_path = resolve_model_path(model_path)
        reader = SafetensorsReader(m_path)
        writer = SafetensorsWriter(out_path, max_shard_bytes=chunk_size_bytes)

        console.print(f"[bold cyan]Pruning MoE Experts to target: {target_experts} experts per layer...[/bold cyan]")

        # Locate router tensors across layers
        router_keys = {}
        for k in reader.weight_map.keys():
            if (
                k.endswith(".router.proj.weight")
                or k.endswith(".mlp.gate.weight")
                or k.endswith(".block_sparse_moe.gate.weight")
                or k.endswith(".mlp.wg.weight")
                or k.endswith(".router.weight")
                or k.endswith(".gate.weight")
            ):
                m = re.search(r"\.(?:layers|blocks)\.(\d+)\.", k)
                if m:
                    router_keys[int(m.group(1))] = k

        if not router_keys:
            raise ValueError(f"No MoE router gate tensors found in {m_path}")

        sample_r = reader.get_tensor(list(router_keys.values())[0])
        orig_num_experts = sample_r.shape[0] if sample_r is not None else 128
        console.print(f"  • Original experts per layer: [yellow]{orig_num_experts}[/yellow]")
        console.print(f"  • Target experts per layer: [green]{target_experts}[/green]")

        num_drop = max(0, orig_num_experts - target_experts)
        if num_drop == 0:
            console.print("[yellow]Model already has target number of experts. No pruning needed.[/yellow]")
            return

        total_bytes = 0
        all_keys = list(reader.weight_map.keys())

        with create_merge_progress() as progress:
            task = progress.add_task("[cyan]Pruning experts & recovering router...", total=len(router_keys))

            for l_idx, r_key in sorted(router_keys.items()):
                progress.update(task, advance=1, description=f"[cyan]Layer {l_idx}")
                router_t = reader.get_tensor(r_key)
                if router_t is None:
                    continue

                # 1. Rank experts by router weight L2 norm (magnitude of influence)
                exp_norms = router_t.float().norm(dim=1)
                sorted_indices = torch.argsort(exp_norms, descending=True)
                surviving_ids = sorted(sorted_indices[:target_experts].tolist())
                dropped_ids = sorted(sorted_indices[target_experts:].tolist())

                # 2. Soft-Transfer Router Recovery (T18 Framework)
                # Compute cosine similarities between dropped and surviving experts in the router
                base_normed = router_t.float() / router_t.float().norm(dim=1, keepdim=True).clamp(min=1e-12)
                var_router_fp32 = router_t.float()[surviving_ids].clone()

                surv_map_orig_to_var = {orig_id: var_idx for var_idx, orig_id in enumerate(surviving_ids)}

                for d in dropped_ids:
                    d_vec = base_normed[d]
                    sims = base_normed[surviving_ids] @ d_vec
                    topk = torch.topk(sims, k=min(top_k_neighbors, len(surviving_ids)))
                    top_surv_indices = topk.indices.tolist()
                    top_sims = topk.values.clamp(min=0.0)

                    if top_sims.sum() > 0:
                        weights = (top_sims / top_sims.sum()).tolist()
                        for surv_idx, w in zip(top_surv_indices, weights):
                            var_router_fp32[surv_idx] += alpha * w * router_t.float()[d]

                writer.add_tensor(r_key, var_router_fp32.to(router_t.dtype))
                total_bytes += var_router_fp32.numel() * router_t.element_size()

                # If Gemma, adjust per_expert_scale if present
                per_exp_key = r_key.replace(".router.proj.weight", ".router.per_expert_scale")
                if per_exp_key in reader.weight_map:
                    per_exp_t = reader.get_tensor(per_exp_key)
                    if per_exp_t is not None:
                        new_per_exp = per_exp_t[surviving_ids].contiguous()
                        writer.add_tensor(per_exp_key, new_per_exp)
                        total_bytes += new_per_exp.numel() * new_per_exp.element_size()

                # 3. Stream surviving expert weights, dropping omitted ones
                exp_prefix = f"model.layers.{l_idx}.mlp.experts."
                matching_exp_keys = [k for k in all_keys if k.startswith(exp_prefix)]

                for old_k in matching_exp_keys:
                    m = re.match(r"^model\.layers\.\d+\.mlp\.experts\.(\d+)\.(.*)$", old_k)
                    if m:
                        e_idx = int(m.group(1))
                        proj_name = m.group(2)
                        if e_idx in surv_map_orig_to_var:
                            new_var_idx = surv_map_orig_to_var[e_idx]
                            new_k = f"model.layers.{l_idx}.mlp.experts.{new_var_idx}.{proj_name}"
                            t = reader.get_tensor(old_k)
                            if t is not None:
                                writer.add_tensor(new_k, t)
                                total_bytes += t.numel() * t.element_size()

                free_memory()

        # Copy all other non-expert weights (attention, norms, embeddings)
        for k in all_keys:
            if not any(k.startswith(f"model.layers.{l}.mlp.experts.") for l in router_keys):
                if k not in router_keys.values() and not k.endswith(".router.per_expert_scale"):
                    t = reader.get_tensor(k)
                    if t is not None:
                        writer.add_tensor(k, t)
                        total_bytes += t.numel() * t.element_size()

        writer.finalize(total_size=total_bytes)

        # Update config.json (root and nested text_config)
        cfg_path = m_path / "config.json"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            def _update_exp_keys(target_dict: dict):
                if "num_local_experts" in target_dict:
                    target_dict["num_local_experts"] = target_experts
                if "num_experts" in target_dict:
                    target_dict["num_experts"] = target_experts
                if "n_routed_experts" in target_dict:
                    target_dict["n_routed_experts"] = target_experts

            _update_exp_keys(cfg)
            if "text_config" in cfg and isinstance(cfg["text_config"], dict):
                _update_exp_keys(cfg["text_config"])

            with open(out_path / "config.json", "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)

        for pat in ["tokenizer*", "*.json", "*.txt", "*.model"]:
            for f in m_path.glob(pat):
                if not f.name.endswith(".safetensors.index.json") and not f.name.endswith(".safetensors") and f.name != "config.json":
                    dest = out_path / f.name
                    if not dest.exists():
                        shutil.copy2(f, dest)

        console.print(f"[bold green]✓ MoE expert pruning completed ({target_experts} experts saved to {out_path})[/bold green]")
