from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

class GGUFReaderWrapper:
    def __init__(self, gguf_path: Path):
        self.gguf_path = gguf_path
        from gguf import GGUFReader
        self.reader = GGUFReader(str(gguf_path))
        self.tensor_info: Dict[str, Any] = {t.name: t for t in self.reader.tensors}

    @property
    def tensor_names(self) -> List[str]:
        return list(self.tensor_info.keys())

    def get_tensor_f32(self, name: str) -> Optional[np.ndarray]:
        t_info = self.tensor_info.get(name)
        if t_info is None:
            return None
        from gguf.quants import dequantize
        arr = dequantize(t_info.data, t_info.tensor_type)
        return arr.reshape(t_info.shape)

    def get_tensor_type(self, name: str):
        t_info = self.tensor_info.get(name)
        return t_info.tensor_type if t_info else None

class GGUFWriterWrapper:
    def __init__(self, output_path: Path, arch_name: str = "llama"):
        self.output_path = output_path
        from gguf import GGUFWriter
        self.writer = GGUFWriter(str(output_path), arch_name)

    def add_tensor(self, name: str, data: np.ndarray, raw_dtype=None):
        if raw_dtype is not None:
            self.writer.add_tensor(name, data, raw_dtype=raw_dtype)
        else:
            self.writer.add_tensor(name, data)

    def write(self):
        self.writer.write_header_to_file()
        self.writer.write_kv_data_to_file()
        self.writer.write_tensors_to_file()
        self.writer.close()
