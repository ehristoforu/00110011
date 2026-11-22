import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from polymerge.io.safetensors_io import SafetensorsReader, SafetensorsWriter, resolve_model_path
from polymerge.utils import human_bytes, free_memory
from polymerge.ui.console import console, create_merge_progress

VOCAB_TENSOR_NAMES = [
    "model.embed_tokens.weight",
    "lm_head.weight",
    "transformer.wte.weight",
    "word_embeddings.weight",
]

class TokenizerTool:
    @staticmethod
    def align_vocabularies(
        base_model: str,
        source_models: List[str],
        output_dir: str,
        pad_to_multiple_of: Optional[int] = 64,
    ):
        import torch

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        base_path = resolve_model_path(base_model)
        source_paths = [resolve_model_path(p) for p in source_models]

        base_reader = SafetensorsReader(base_path)
        source_readers = [SafetensorsReader(p) for p in source_paths]

        console.print(f"[bold cyan]Aligning vocabularies across {len(source_models) + 1} models...[/bold cyan]")

        # Find vocabulary sizes from embed_tokens or lm_head
        vocab_sizes = []
        for r in [base_reader] + source_readers:
            sz = None
            for name in VOCAB_TENSOR_NAMES:
                if name in r.weight_map:
                    t = r.get_tensor(name)
                    if t is not None:
                        sz = t.shape[0]
                        break
            vocab_sizes.append(sz or 32000)

        target_vocab_size = max(vocab_sizes)
        if pad_to_multiple_of and target_vocab_size % pad_to_multiple_of != 0:
            target_vocab_size = ((target_vocab_size // pad_to_multiple_of) + 1) * pad_to_multiple_of

        console.print(f"  • Base vocab: [yellow]{vocab_sizes[0]}[/yellow]")
        console.print(f"  • Source vocabs: [yellow]{vocab_sizes[1:]}[/yellow]")
        console.print(f"  • Target aligned vocab: [bold green]{target_vocab_size}[/bold green]")

        writer = SafetensorsWriter(out_path)
        total_bytes = 0

        with create_merge_progress() as progress:
            task = progress.add_task("[cyan]Resizing & padding embedding layers...", total=len(base_reader.weight_map))

            for key in base_reader.weight_map.keys():
                progress.update(task, advance=1, description=f"[cyan]{key[:35]}")
                t = base_reader.get_tensor(key)
                if t is None:
                    continue

                if key in VOCAB_TENSOR_NAMES and t.shape[0] < target_vocab_size:
                    # Pad vocabulary dimension with zeros or mean
                    pad_rows = target_vocab_size - t.shape[0]
                    padding = torch.zeros((pad_rows, *t.shape[1:]), dtype=t.dtype, device=t.device)
                    t_aligned = torch.cat([t, padding], dim=0)
                    writer.add_tensor(key, t_aligned)
                    total_bytes += t_aligned.numel() * t_aligned.element_size()
                else:
                    writer.add_tensor(key, t)
                    total_bytes += t.numel() * t.element_size()

                free_memory()

        writer.finalize(total_size=total_bytes)

        # Update config.json vocab_size
        cfg_path = base_path / "config.json"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg["vocab_size"] = target_vocab_size
            with open(out_path / "config.json", "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)

        # Copy tokenizer assets from largest vocabulary model or base
        for pat in ["tokenizer*", "*.json", "*.txt", "*.model"]:
            for f in base_path.glob(pat):
                if not f.name.endswith(".safetensors.index.json") and not f.name.endswith(".safetensors") and f.name != "config.json":
                    dest = out_path / f.name
                    if not dest.exists():
                        shutil.copy2(f, dest)

        console.print(f"[bold green]✓ Vocabulary alignment complete! ({human_bytes(total_bytes)})[/bold green]")

    @staticmethod
    def truncate_vocab(
        model_path: str,
        output_dir: str,
        target_vocab_size: int,
    ):
        import torch

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        m_path = resolve_model_path(model_path)
        reader = SafetensorsReader(m_path)
        writer = SafetensorsWriter(out_path)

        console.print(f"[bold cyan]Truncating vocabulary of {m_path} to {target_vocab_size}...[/bold cyan]")
        total_bytes = 0

        with create_merge_progress() as progress:
            task = progress.add_task("[cyan]Truncating vocabulary tensors...", total=len(reader.weight_map))

            for key in reader.weight_map.keys():
                progress.update(task, advance=1, description=f"[cyan]{key[:35]}")
                t = reader.get_tensor(key)
                if t is None:
                    continue

                if key in VOCAB_TENSOR_NAMES and t.shape[0] > target_vocab_size:
                    t_trunc = t[:target_vocab_size].contiguous()
                    writer.add_tensor(key, t_trunc)
                    total_bytes += t_trunc.numel() * t_trunc.element_size()
                else:
                    writer.add_tensor(key, t)
                    total_bytes += t.numel() * t.element_size()

                free_memory()

        writer.finalize(total_size=total_bytes)

        cfg_path = m_path / "config.json"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg["vocab_size"] = target_vocab_size
            with open(out_path / "config.json", "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)

        for pat in ["tokenizer*", "*.json", "*.txt", "*.model"]:
            for f in m_path.glob(pat):
                if not f.name.endswith(".safetensors.index.json") and not f.name.endswith(".safetensors") and f.name != "config.json":
                    dest = out_path / f.name
                    if not dest.exists():
                        shutil.copy2(f, dest)

        console.print(f"[bold green]✓ Vocabulary truncated to {target_vocab_size}![/bold green]")
