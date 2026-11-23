import os
import json
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

def resolve_model_path(model_id_or_path: str, cache_dir: Optional[str] = None) -> Path:
    p = Path(model_id_or_path)
    if p.exists() and p.is_dir():
        return p
    from huggingface_hub import snapshot_download
    from polymerge.utils import setup_fast_hf_transfer
    setup_fast_hf_transfer()
    local_dir = snapshot_download(
        repo_id=model_id_or_path,
        cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
        ignore_patterns=["*.bin", "*.pt"],
    )
    return Path(local_dir)

def load_weight_map(model_dir: Path) -> Tuple[Dict[str, str], List[str]]:
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        weight_map = data.get("weight_map", {})
        shards = sorted(list(set(weight_map.values())))
        return weight_map, shards
    single_files = sorted(glob.glob(str(model_dir / "*.safetensors")))
    if not single_files:
        raise FileNotFoundError(f"No safetensors found in {model_dir}")
    from safetensors import safe_open
    weight_map = {}
    shards = []
    for sf in single_files:
        fname = Path(sf).name
        shards.append(fname)
        with safe_open(str(model_dir / fname), framework="pt", device="cpu") as f:
            for k in f.keys():
                weight_map[k] = fname
    return weight_map, shards

class SafetensorsReader:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.weight_map, self.shards = load_weight_map(model_dir)
        self._handles: Dict[str, Any] = {}

    def get_tensor(self, name: str, device: str = "cpu"):
        from safetensors import safe_open
        shard = self.weight_map.get(name)
        if not shard:
            return None
        if shard not in self._handles:
            self._handles[shard] = safe_open(str(self.model_dir / shard), framework="pt", device="cpu")
        handle = self._handles[shard]
        t = handle.get_tensor(name)
        if device != "cpu":
            t = t.to(device)
        return t

    def close(self):
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class SafetensorsWriter:
    def __init__(self, output_dir: Path, max_shard_bytes: int = 4_000_000_000):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_shard_bytes = max_shard_bytes
        self.current_shard_tensors: Dict[str, Any] = {}
        self.current_shard_bytes = 0
        self.shard_idx = 1
        self.shards_written: List[str] = []
        self.weight_map: Dict[str, str] = {}

    def add_tensor(self, name: str, tensor):
        from safetensors.torch import save_file
        num_bytes = tensor.numel() * tensor.element_size()
        if self.current_shard_bytes + num_bytes > self.max_shard_bytes and self.current_shard_tensors:
            self.flush()
        self.current_shard_tensors[name] = tensor.contiguous()
        self.current_shard_bytes += num_bytes

    def flush(self):
        if not self.current_shard_tensors:
            return
        from safetensors.torch import save_file
        from polymerge.utils import free_memory
        shard_name = f"model-{self.shard_idx:05d}-of-XXXXX.safetensors"
        out_path = self.output_dir / shard_name
        save_file(self.current_shard_tensors, str(out_path), metadata={"format": "pt"})
        self.shards_written.append(shard_name)
        for name in self.current_shard_tensors:
            self.weight_map[name] = shard_name
        self.current_shard_tensors.clear()
        self.current_shard_bytes = 0
        self.shard_idx += 1
        free_memory()

    def finalize(self, total_size: int = 0):
        self.flush()
        total_shards = len(self.shards_written)
        if total_shards == 0:
            return
        if total_shards == 1:
            old_name = self.shards_written[0]
            new_name = "model.safetensors"
            os.rename(self.output_dir / old_name, self.output_dir / new_name)
            self.weight_map = {k: new_name for k in self.weight_map}
            return
        new_weight_map = {}
        for idx, old_name in enumerate(self.shards_written, start=1):
            new_name = f"model-{idx:05d}-of-{total_shards:05d}.safetensors"
            os.rename(self.output_dir / old_name, self.output_dir / new_name)
            for k, v in self.weight_map.items():
                if v == old_name:
                    new_weight_map[k] = new_name
        index_data = {
            "metadata": {"total_size": total_size},
            "weight_map": new_weight_map,
        }
        with open(self.output_dir / "model.safetensors.index.json", "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)
