import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from polymerge.io.safetensors_io import SafetensorsReader, SafetensorsWriter, resolve_model_path
from polymerge.utils import human_bytes, free_memory
from polymerge.ui.console import console, create_merge_progress

class LayerSlicer:
    def __init__(self, slices: List[Dict[str, Any]], chunk_size_bytes: int = 2_000_000_000):
        self.slices = slices
        self.chunk_size_bytes = chunk_size_bytes

    def execute(self, output_dir: str):
        import torch

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        writer = SafetensorsWriter(out_path, max_shard_bytes=self.chunk_size_bytes)

        model_readers: Dict[str, SafetensorsReader] = {}
        for s in self.slices:
            for src in s.get("sources", []):
                m_path = src.get("model")
                if m_path and m_path not in model_readers:
                    resolved = resolve_model_path(m_path)
                    model_readers[m_path] = SafetensorsReader(resolved)

        first_model_reader = list(model_readers.values())[0]

        console.print(f"[bold cyan]Executing Layer Slicing & Stacking into {len(self.slices)} slices...[/bold cyan]")
        total_layers = 0
        total_bytes = 0

        # Copy non-layer weights from the primary model
        for name in first_model_reader.weight_map.keys():
            if not any(f".layers." in name for _ in [1]) and not any(f".blocks." in name for _ in [1]):
                t = first_model_reader.get_tensor(name)
                if t is not None:
                    writer.add_tensor(name, t)
                    total_bytes += t.numel() * t.element_size()

        # Iterate through slice definitions and remap layer indices
        target_layer_idx = 0
        with create_merge_progress() as progress:
            task = progress.add_task("[cyan]Slicing layers...", total=len(self.slices))

            for s_idx, s in enumerate(self.slices):
                progress.update(task, advance=1, description=f"[cyan]Slice {s_idx + 1}")
                sources = s.get("sources", [])
                for src in sources:
                    m_name = src.get("model")
                    l_start, l_end = src.get("layer_range", [0, 0])
                    reader = model_readers[m_name]

                    for source_l in range(l_start, l_end):
                        layer_prefix = f"model.layers.{source_l}."
                        matching = [k for k in reader.weight_map.keys() if k.startswith(layer_prefix)]

                        for k in matching:
                            sub_name = k[len(layer_prefix):]
                            new_name = f"model.layers.{target_layer_idx}.{sub_name}"
                            t = reader.get_tensor(k)
                            if t is not None:
                                writer.add_tensor(new_name, t)
                                total_bytes += t.numel() * t.element_size()

                        target_layer_idx += 1
                        free_memory()

        writer.finalize(total_size=total_bytes)

        # Update config.json num_hidden_layers (both root and nested text_config)
        base_dir = first_model_reader.model_dir
        cfg_path = base_dir / "config.json"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg["num_hidden_layers"] = target_layer_idx
            if "text_config" in cfg and isinstance(cfg["text_config"], dict):
                cfg["text_config"]["num_hidden_layers"] = target_layer_idx
            with open(out_path / "config.json", "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)

        for pat in ["tokenizer*", "*.txt", "*.model"]:
            for f in base_dir.glob(pat):
                dest = out_path / f.name
                if not dest.exists():
                    import shutil
                    shutil.copy2(f, dest)

        console.print(f"[bold green]✓ Layer Slicing Completed! Target layers: {target_layer_idx} ({human_bytes(total_bytes)})[/bold green]")
