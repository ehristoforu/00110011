import os
import shutil
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch

from polymerge.config import MergeConfig
from polymerge.filter import TensorFilter
from polymerge.methods import get_merge_method
from polymerge.utils import select_device, free_memory, human_bytes
from polymerge.io.safetensors_io import resolve_model_path, SafetensorsReader, SafetensorsWriter
from polymerge.ui.console import console, print_banner, print_plan_table, create_merge_progress

class MergeEngine:
    def __init__(self, config: MergeConfig):
        self.config = config
        self.device = select_device(config.device)
        self.tensor_filter = TensorFilter(
            ignore_vision=config.ignore_vision,
            custom_ignore_patterns=config.ignore_patterns,
            skip_patterns=config.skip_patterns,
        )
        self.merge_method = get_merge_method(config.merge_method)

    def run(self, output_dir: str):
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if self.config.staged and self.config.stages:
            self._run_staged(out_path)
        else:
            self._run_direct(out_path)

    def _run_staged(self, final_output_dir: Path):
        stages = self.config.stages
        console.print(f"[bold yellow]Executing {len(stages)} Stages Workflow...[/bold yellow]")
        prev_output = None
        for i, stage_dict in enumerate(stages, start=1):
            console.print(f"\n[bold magenta]=== Stage {i}/{len(stages)}: {stage_dict.get('name', f'Stage {i}')} ===[/bold magenta]")
            stage_out = final_output_dir / f"stage_{i}" if i < len(stages) else final_output_dir
            stage_out.mkdir(parents=True, exist_ok=True)

            stage_cfg_data = dict(stage_dict)
            if prev_output and stage_cfg_data.get("use_previous_output_as_base", False):
                stage_cfg_data["base_model"] = str(prev_output)

            models_raw = stage_cfg_data.get("models", [])
            for m in models_raw:
                if isinstance(m, dict) and m.get("model") == "PREVIOUS_STAGE":
                    m["model"] = str(prev_output)

            stage_cfg = MergeConfig.from_yaml(yaml.dump(stage_cfg_data)) if "yaml" in globals() else MergeConfig(
                merge_method=stage_cfg_data.get("merge_method", self.config.merge_method),
                base_model=stage_cfg_data.get("base_model", self.config.base_model),
                dtype=stage_cfg_data.get("dtype", self.config.dtype),
                ignore_vision=stage_cfg_data.get("ignore_vision", self.config.ignore_vision),
                skip_patterns=stage_cfg_data.get("skip_patterns", self.config.skip_patterns),
                device=self.config.device,
            )
            sub_engine = MergeEngine(stage_cfg)
            sub_engine._run_direct(stage_out)
            prev_output = stage_out
            free_memory()

    def _run_direct(self, out_path: Path):
        print_banner()
        print_plan_table(
            merge_method=self.config.merge_method,
            base_model=self.config.base_model,
            models=self.config.models,
            device=str(self.device),
            out_format=self.config.out_format,
        )

        base_dir: Optional[Path] = None
        base_reader: Optional[SafetensorsReader] = None
        if self.config.base_model:
            with console.status(f"[bold green]Locating base model: {self.config.base_model}..."):
                base_dir = resolve_model_path(self.config.base_model)
                base_reader = SafetensorsReader(base_dir)

        source_readers: List[SafetensorsReader] = []
        source_dirs: List[Path] = []
        for sm in self.config.models:
            with console.status(f"[bold green]Locating source: {sm.model}..."):
                s_dir = resolve_model_path(sm.model)
                source_dirs.append(s_dir)
                source_readers.append(SafetensorsReader(s_dir))

        all_tensor_names = []
        if base_reader:
            all_tensor_names.extend(base_reader.weight_map.keys())
        for sr in source_readers:
            for k in sr.weight_map.keys():
                if k not in all_tensor_names:
                    all_tensor_names.append(k)

        weights = [m.weight for m in self.config.models]
        densities = [m.density for m in self.config.models]

        writer = SafetensorsWriter(out_path, max_shard_bytes=self.config.chunk_size_bytes)
        total_processed_bytes = 0
        ignored_count = 0
        merged_count = 0
        copied_count = 0

        with create_merge_progress() as progress:
            task = progress.add_task("[cyan]Merging tensors...", total=len(all_tensor_names))

            for name in all_tensor_names:
                progress.update(task, advance=1, description=f"[cyan]Tensor: [white]{name[:35]}")

                if self.tensor_filter.should_ignore(name):
                    ignored_count += 1
                    continue

                base_t = base_reader.get_tensor(name) if base_reader else None

                if self.tensor_filter.should_skip_merge(name):
                    if base_t is not None:
                        writer.add_tensor(name, base_t)
                        total_processed_bytes += base_t.numel() * base_t.element_size()
                        copied_count += 1
                    continue

                if base_t is not None and (not base_t.is_floating_point()):
                    writer.add_tensor(name, base_t)
                    total_processed_bytes += base_t.numel() * base_t.element_size()
                    copied_count += 1
                    continue

                present_sources = []
                present_weights = []
                present_densities = []

                for idx, sr in enumerate(source_readers):
                    t = sr.get_tensor(name)
                    if t is not None:
                        if self.device.type != "cpu":
                            t = t.to(self.device)
                        present_sources.append(t)
                        present_weights.append(weights[idx])
                        present_densities.append(densities[idx])

                if not present_sources:
                    if base_t is not None:
                        writer.add_tensor(name, base_t)
                        total_processed_bytes += base_t.numel() * base_t.element_size()
                        copied_count += 1
                    continue

                if base_t is not None and self.device.type != "cpu":
                    base_t_dev = base_t.to(self.device)
                else:
                    base_t_dev = base_t

                if len(present_sources) == 1 and base_t is None:
                    merged = present_sources[0]
                else:
                    merged = self.merge_method.merge(
                        base_tensor=base_t_dev,
                        source_tensors=present_sources,
                        weights=present_weights,
                        densities=present_densities,
                        parameters=self.config.parameters,
                    )

                if self.config.dtype == "bfloat16":
                    merged = merged.to(torch.bfloat16)
                elif self.config.dtype == "float16":
                    merged = merged.to(torch.float16)
                elif self.config.dtype == "float32":
                    merged = merged.to(torch.float32)

                merged_cpu = merged.to("cpu")
                writer.add_tensor(name, merged_cpu)
                total_processed_bytes += merged_cpu.numel() * merged_cpu.element_size()
                merged_count += 1

                del present_sources, base_t, base_t_dev, merged, merged_cpu

        with console.status("[bold green]Finalizing shards and metadata..."):
            writer.finalize(total_size=total_processed_bytes)
            self._copy_metadata(base_dir or source_dirs[0], out_path)

        console.print(f"\n[bold green]✓ Merge Completed Successfully![/bold green]")
        console.print(f"  • Total tensors merged: [bold]{merged_count}[/bold]")
        console.print(f"  • Non-mergeable / base preserved: [bold]{copied_count}[/bold]")
        console.print(f"  • Ignored (vision / custom filters): [bold]{ignored_count}[/bold]")
        console.print(f"  • Total model size: [bold]{human_bytes(total_processed_bytes)}[/bold]")
        console.print(f"  • Output saved to: [cyan]{out_path.absolute()}[/cyan]\n")

    def _copy_metadata(self, source_dir: Path, out_path: Path):
        out_path.mkdir(parents=True, exist_ok=True)
        for pattern in ["*.json", "*.txt", "*.model", "tokenizer*"]:
            for f in source_dir.glob(pattern):
                if not f.name.endswith(".safetensors.index.json") and not f.name.endswith(".safetensors"):
                    dest = out_path / f.name
                    if not dest.exists():
                        shutil.copy2(f, dest)

        # If vision was ignored, sanitize config.json to prevent HF/vLLM from expecting missing vision weights
        if self.config.ignore_vision:
            out_cfg_path = out_path / "config.json"
            if out_cfg_path.exists():
                try:
                    with open(out_cfg_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)

                    modified = False
                    # Remove vision_config if visual weights were stripped
                    if "vision_config" in cfg:
                        del cfg["vision_config"]
                        modified = True

                    # Demote VLM architecture to text-only CausalLM if applicable
                    archs = cfg.get("architectures", [])
                    new_archs = []
                    for arch in archs:
                        if "ConditionalGeneration" in arch or "VL" in arch or "Vision" in arch:
                            if "Qwen2VL" in arch or "Qwen2_5_VL" in arch:
                                new_archs.append("Qwen2ForCausalLM")
                            elif "Qwen3_5Moe" in arch:
                                new_archs.append("Qwen3_5MoeForCausalLM")
                            elif "Llama" in arch:
                                new_archs.append("LlamaForCausalLM")
                            else:
                                new_archs.append(arch.replace("ForConditionalGeneration", "ForCausalLM"))
                            modified = True
                        else:
                            new_archs.append(arch)

                    if modified:
                        cfg["architectures"] = new_archs
                        # If model_type is VL, demote to base text type
                        m_type = cfg.get("model_type", "")
                        if "vl" in m_type:
                            cfg["model_type"] = m_type.replace("_vl", "").replace("-vl", "")
                        with open(out_cfg_path, "w", encoding="utf-8") as f:
                            json.dump(cfg, f, indent=2)
                except Exception:
                    pass
