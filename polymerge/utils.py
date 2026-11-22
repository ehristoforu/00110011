import os
import shutil
import gc
from typing import Optional

def setup_fast_hf_transfer():
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def select_device(requested: str = "auto"):
    req = requested.lower()
    try:
        import torch
    except ImportError:
        # Fallback device representation when torch is not installed in bare test runner
        class FallbackDevice:
            def __init__(self, t):
                self.type = t
            def __str__(self):
                return self.type
        return FallbackDevice("cpu" if req in ("auto", "cpu") else req)

    if req == "cpu":
        return torch.device("cpu")
    if req in ("gpu", "cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if req == "tpu":
        try:
            import torch_xla.core.xla_model as xm
            return xm.xla_device()
        except ImportError:
            return torch.device("cpu")
    if req == "auto":
        try:
            import torch_xla.core.xla_model as xm
            return xm.xla_device()
        except Exception:
            pass
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)

def free_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        os.sync()
        if os.path.exists("/proc/sys/vm/drop_caches"):
            with open("/proc/sys/vm/drop_caches", "w") as f:
                f.write("3\n")
    except Exception:
        pass

def human_bytes(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"
