from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from polymerge.methods import get_merge_method
from polymerge.io.gguf_io import GGUFReaderWrapper, GGUFWriterWrapper
from polymerge.filter import TensorFilter
from polymerge.ui.console import console, create_merge_progress

class GGUFMergeEngine:
    def __init__(
        self,
        base_gguf: Optional[str],
        source_ggufs: List[str],
        weights: List[float],
        densities: List[float],
        merge_method: str = "linear",
        parameters: Optional[dict] = None,
        ignore_vision: bool = True,
    ):
        self.base_gguf = Path(base_gguf) if base_gguf else None
        self.source_ggufs = [Path(p) for p in source_ggufs]
        self.weights = weights
        self.densities = densities
        self.method = get_merge_method(merge_method)
        self.parameters = parameters or {}
        self.filter = TensorFilter(ignore_vision=ignore_vision)

    def merge_to_gguf(self, output_gguf_path: str):
        import torch

        out_path = Path(output_gguf_path)
        base_reader = GGUFReaderWrapper(self.base_gguf) if self.base_gguf else None
        source_readers = [GGUFReaderWrapper(p) for p in self.source_ggufs]

        all_names = []
        if base_reader:
            all_names.extend(base_reader.tensor_names)
        for sr in source_readers:
            for n in sr.tensor_names:
                if n not in all_names:
                    all_names.append(n)

        writer = GGUFWriterWrapper(out_path)

        with create_merge_progress() as progress:
            task = progress.add_task("[yellow]Merging GGUF quantized models...", total=len(all_names))
            for name in all_names:
                progress.update(task, advance=1, description=f"[yellow]GGUF: [white]{name[:35]}")
                if self.filter.should_ignore(name):
                    continue

                base_np = base_reader.get_tensor_f32(name) if base_reader else None
                base_t = torch.from_numpy(base_np) if base_np is not None else None

                sources_t = []
                active_weights = []
                active_densities = []

                for idx, sr in enumerate(source_readers):
                    s_np = sr.get_tensor_f32(name)
                    if s_np is not None:
                        sources_t.append(torch.from_numpy(s_np))
                        active_weights.append(self.weights[idx])
                        active_densities.append(self.densities[idx])

                if not sources_t:
                    if base_t is not None:
                        writer.add_tensor(name, base_np)
                    continue

                if len(sources_t) == 1 and base_t is None:
                    merged_t = sources_t[0]
                else:
                    merged_t = self.method.merge(
                        base_tensor=base_t,
                        source_tensors=sources_t,
                        weights=active_weights,
                        densities=active_densities,
                        parameters=self.parameters,
                    )

                writer.add_tensor(name, merged_t.numpy())
                del sources_t, base_t, merged_t

        writer.write()
        console.print(f"[bold green]✓ GGUF Merged successfully saved to {out_path}[/bold green]")
