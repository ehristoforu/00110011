import re
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json

EXPERT_PATTERNS = [
    re.compile(r"^(.*(?:\.mlp|\.feed_forward|\.block_sparse_moe))\.(gate_proj|up_proj|down_proj)\.weight$"),
    re.compile(r"^(.*(?:\.mlp|\.feed_forward|\.block_sparse_moe))\.(w1|w2|w3)\.weight$"),
    re.compile(r"^(.*(?:\.mlp|\.feed_forward))\.(fc1|fc2|c_fc|c_proj)\.weight$"),
]

def detect_dense_mlp(tensor_name: str) -> Optional[Tuple[str, str]]:
    for pat in EXPERT_PATTERNS:
        m = pat.match(tensor_name)
        if m:
            prefix, proj = m.group(1), m.group(2)
            if proj in ("gate_proj", "w1", "fc1", "c_fc"):
                std_proj = "w1"
            elif proj in ("down_proj", "w2", "fc2", "c_proj"):
                std_proj = "w2"
            else:
                std_proj = "w3"
            return prefix, std_proj
    return None

class MoEAssembler:
    def __init__(
        self,
        base_model: str,
        expert_models: List[str],
        gate_mode: str = "uniform_random",
        target_format: str = "mixtral",
        shared_expert_model: Optional[str] = None,
        pack_experts: bool = False,
        chunk_size_bytes: int = 2_000_000_000,
    ):
        from polymerge.io.safetensors_io import resolve_model_path
        self.base_model_path = resolve_model_path(base_model)
        self.expert_paths = [resolve_model_path(m) for m in expert_models]
        self.gate_mode = gate_mode
        self.target_format = target_format.lower()
        self.shared_expert_path = resolve_model_path(shared_expert_model) if shared_expert_model else None
        self.pack_experts = pack_experts
        self.chunk_size_bytes = chunk_size_bytes

    def assemble(self, output_dir: str):
        import torch
        from polymerge.io.safetensors_io import SafetensorsReader, SafetensorsWriter
        from polymerge.utils import human_bytes, free_memory
        from polymerge.ui.console import console, create_merge_progress

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        base_reader = SafetensorsReader(self.base_model_path)
        expert_readers = [SafetensorsReader(p) for p in self.expert_paths]
        shared_reader = SafetensorsReader(self.shared_expert_path) if self.shared_expert_path else None
        writer = SafetensorsWriter(out_path, max_shard_bytes=self.chunk_size_bytes)

        console.print(f"[bold cyan]Assembling MoE from {len(expert_readers)} experts...[/bold cyan]")
        console.print(f"  • Base model: [white]{self.base_model_path}[/white]")
        console.print(f"  • Target format: [green]{self.target_format}[/green]")
        if self.shared_expert_path:
            console.print(f"  • Shared expert: [yellow]{self.shared_expert_path}[/yellow]")

        base_tensors = list(base_reader.weight_map.keys())
        processed_mlp_prefixes = set()
        total_bytes = 0

        with create_merge_progress() as progress:
            task = progress.add_task("[cyan]Processing base & routing...", total=len(base_tensors))

            for name in base_tensors:
                progress.update(task, advance=1, description=f"[cyan]{name[:35]}")
                mlp_info = detect_dense_mlp(name)

                if mlp_info is None:
                    t = base_reader.get_tensor(name)
                    if t is not None:
                        writer.add_tensor(name, t)
                        total_bytes += t.numel() * t.element_size()
                    continue

                prefix, _ = mlp_info
                if prefix in processed_mlp_prefixes:
                    continue
                processed_mlp_prefixes.add(prefix)

                num_experts = len(expert_readers)
                sample_t = base_reader.get_tensor(name)
                hidden_size = sample_t.shape[-1] if sample_t is not None else 4096

                if self.target_format == "gemma":
                    gate_proj_name = f"{prefix}.router.proj.weight"
                    scale_name = f"{prefix}.router.scale"
                    per_exp_name = f"{prefix}.router.per_expert_scale"

                    if self.gate_mode == "uniform_random":
                        scale_val = (1.0 / hidden_size) ** 0.5
                        gate_weight = (torch.rand(num_experts, hidden_size, dtype=torch.float32) * 2 * scale_val - scale_val).to(torch.bfloat16)
                    else:
                        gate_weight = torch.randn(num_experts, hidden_size, dtype=torch.bfloat16) * 0.02

                    writer.add_tensor(gate_proj_name, gate_weight)
                    total_bytes += gate_weight.numel() * gate_weight.element_size()

                    r_scale = torch.ones(hidden_size, dtype=torch.bfloat16)
                    writer.add_tensor(scale_name, r_scale)
                    total_bytes += r_scale.numel() * r_scale.element_size()

                    per_exp_scale = torch.ones(num_experts, dtype=torch.bfloat16)
                    writer.add_tensor(per_exp_name, per_exp_scale)
                    total_bytes += per_exp_scale.numel() * per_exp_scale.element_size()
                else:
                    gate_name = f"{prefix}.gate.weight"
                    if self.gate_mode == "uniform_random":
                        scale_val = (1.0 / hidden_size) ** 0.5
                        gate_weight = (torch.rand(num_experts, hidden_size, dtype=torch.float32) * 2 * scale_val - scale_val).to(torch.bfloat16)
                    else:
                        gate_weight = torch.randn(num_experts, hidden_size, dtype=torch.bfloat16) * 0.02

                    writer.add_tensor(gate_name, gate_weight)
                    total_bytes += gate_weight.numel() * gate_weight.element_size()

                if shared_reader or self.target_format in ("qwen", "deepseek"):
                    s_source = shared_reader or base_reader
                    for proj_orig, proj_std in [("gate_proj", "w1"), ("down_proj", "w2"), ("up_proj", "w3")]:
                        src_name = f"{prefix}.{proj_orig}.weight"
                        if src_name not in s_source.weight_map:
                            src_name = f"{prefix}.{proj_std}.weight"
                        st = s_source.get_tensor(src_name)
                        if st is not None:
                            shared_name = f"{prefix}.shared_expert.{proj_orig}.weight"
                            writer.add_tensor(shared_name, st)
                            total_bytes += st.numel() * st.element_size()

                for exp_idx, exp_reader in enumerate(expert_readers):
                    for proj_orig, proj_std in [("gate_proj", "w1"), ("down_proj", "w2"), ("up_proj", "w3")]:
                        src_name = f"{prefix}.{proj_orig}.weight"
                        if src_name not in exp_reader.weight_map:
                            src_name = f"{prefix}.{proj_std}.weight"

                        exp_t = exp_reader.get_tensor(src_name)
                        if exp_t is None:
                            exp_t = base_reader.get_tensor(src_name)

                        if exp_t is not None:
                            if self.target_format == "mixtral":
                                out_name = f"{prefix}.experts.{exp_idx}.{proj_std}.weight"
                            else:
                                out_name = f"{prefix}.experts.{exp_idx}.{proj_orig}.weight"
                            writer.add_tensor(out_name, exp_t)
                            total_bytes += exp_t.numel() * exp_t.element_size()

                free_memory()

        writer.finalize(total_size=total_bytes)
        self._update_moe_config(out_path, len(expert_readers))
        console.print(f"[bold green]✓ Universal MoE Assembled ({human_bytes(total_bytes)})[/bold green]")

    def _update_moe_config(self, out_path: Path, num_experts: int):
        cfg_path = self.base_model_path / "config.json"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            is_vl = "vision_config" in cfg or "qwen3_vl" in cfg.get("model_type", "") or "qwen3_5" in str(cfg.get("model_type", "")) or "image_token_id" in cfg
            sub_text_cfg = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else None

            if self.target_format == "mixtral":
                cfg["model_type"] = "mixtral"
                cfg["architectures"] = ["MixtralForCausalLM"]
                cfg["num_local_experts"] = num_experts
                cfg["num_experts_per_tok"] = min(2, num_experts)
                if sub_text_cfg:
                    sub_text_cfg["num_local_experts"] = num_experts
                    sub_text_cfg["num_experts_per_tok"] = min(2, num_experts)
            elif self.target_format == "qwen":
                if is_vl:
                    # Qwen3-VL / Qwen3.5 Multimodal Architecture
                    cfg["model_type"] = "qwen3_vl_moe" if "qwen3" in str(cfg.get("model_type", "")) or (sub_text_cfg and "qwen3" in str(sub_text_cfg.get("model_type", ""))) else "qwen2_5_vl_moe"
                    cfg["architectures"] = ["Qwen3VLMoEForConditionalGeneration"] if "qwen3" in cfg["model_type"] else ["Qwen2_5_VLMoEForConditionalGeneration"]
                else:
                    cfg["model_type"] = "qwen2_moe"
                    cfg["architectures"] = ["Qwen2MoeForCausalLM"]

                active_experts = min(2, num_experts) if num_experts >= 4 else min(1, num_experts)
                cfg["num_experts"] = num_experts
                cfg["num_experts_per_tok"] = active_experts

                if sub_text_cfg:
                    sub_text_cfg["num_experts"] = num_experts
                    sub_text_cfg["num_experts_per_tok"] = active_experts
            elif self.target_format == "deepseek":
                cfg["model_type"] = "deepseek_v2"
                cfg["architectures"] = ["DeepseekV2ForCausalLM"]
                cfg["n_routed_experts"] = num_experts
                cfg["num_experts_per_tok"] = min(6, num_experts)
                if sub_text_cfg:
                    sub_text_cfg["n_routed_experts"] = num_experts
                    sub_text_cfg["num_experts_per_tok"] = min(6, num_experts)
            elif self.target_format == "gemma":
                cfg["num_local_experts"] = num_experts
                cfg["num_experts_per_tok"] = min(8, num_experts)
                if sub_text_cfg:
                    sub_text_cfg["num_local_experts"] = num_experts
                    sub_text_cfg["num_experts_per_tok"] = min(8, num_experts)
            else:
                cfg["num_local_experts"] = num_experts
                cfg["num_experts_per_tok"] = min(2, num_experts)
                if sub_text_cfg:
                    sub_text_cfg["num_local_experts"] = num_experts
                    sub_text_cfg["num_experts_per_tok"] = min(2, num_experts)

            with open(out_path / "config.json", "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)

        for pat in ["tokenizer*", "*.txt", "*.model"]:
            for f in self.base_model_path.glob(pat):
                dest = out_path / f.name
                if not dest.exists():
                    import shutil
                    shutil.copy2(f, dest)
