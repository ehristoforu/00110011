from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

@dataclass
class SourceModelConfig:
    model: str
    weight: float = 1.0
    density: float = 1.0
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MergeConfig:
    merge_method: str = "linear"
    base_model: Optional[str] = None
    models: List[SourceModelConfig] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    dtype: str = "bfloat16"
    ignore_vision: bool = True
    ignore_patterns: List[str] = field(default_factory=list)
    skip_patterns: List[str] = field(default_factory=list)
    chunk_size_bytes: int = 1_000_000_000
    device: str = "auto"
    num_threads: int = 4
    staged: bool = False
    stages: List[Dict[str, Any]] = field(default_factory=list)
    out_format: str = "safetensors"

    @classmethod
    def from_yaml(cls, path_or_str: str) -> "MergeConfig":
        data = None
        if len(path_or_str) < 260 and not ("\n" in path_or_str):
            p = Path(path_or_str)
            if p.exists() and p.is_file():
                with open(p, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
        if data is None:
            data = yaml.safe_load(path_or_str)

        models_raw = data.get("models", [])
        models = []
        for m in models_raw:
            if isinstance(m, str):
                models.append(SourceModelConfig(model=m))
            elif isinstance(m, dict):
                params = m.get("parameters", {})
                w = m.get("weight", params.get("weight", 1.0))
                d = m.get("density", params.get("density", 1.0))
                models.append(
                    SourceModelConfig(
                        model=m.get("model", ""),
                        weight=float(w) if isinstance(w, (int, float)) else 1.0,
                        density=float(d) if isinstance(d, (int, float)) else 1.0,
                        parameters=params,
                    )
                )

        return cls(
            merge_method=data.get("merge_method", "linear"),
            base_model=data.get("base_model"),
            models=models,
            parameters=data.get("parameters", {}),
            dtype=data.get("dtype", "bfloat16"),
            ignore_vision=data.get("ignore_vision", True),
            ignore_patterns=data.get("ignore_patterns", []),
            skip_patterns=data.get("skip_patterns", []),
            chunk_size_bytes=data.get("chunk_size_bytes", 1_000_000_000),
            device=data.get("device", "auto"),
            num_threads=data.get("num_threads", 4),
            staged=data.get("staged", False),
            stages=data.get("stages", []),
            out_format=data.get("out_format", "safetensors"),
        )
