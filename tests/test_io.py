import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import json
import os

from polymerge.io.safetensors_io import load_weight_map, SafetensorsWriter
from polymerge.io.gguf_io import GGUFReaderWrapper, GGUFWriterWrapper

class TestIOComprehensive(unittest.TestCase):
    def test_load_weight_map_index_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            index_file = tmppath / "model.safetensors.index.json"
            dummy_index = {
                "metadata": {"total_size": 1000},
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                    "model.layers.0.self_attn.q_proj.weight": "model-00002-of-00002.safetensors"
                }
            }
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(dummy_index, f)

            weight_map, shards = load_weight_map(tmppath)
            self.assertEqual(len(weight_map), 2)
            self.assertEqual(shards, ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"])

    def test_load_weight_map_no_files_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            with self.assertRaises(FileNotFoundError):
                load_weight_map(tmppath)

    def test_gguf_wrapper_properties(self):
        mock_gguf = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.name = "blk.0.attn_q.weight"
        mock_tensor.shape = (4096, 4096)
        mock_tensor.tensor_type = 0

        mock_instance = MagicMock()
        mock_instance.tensors = [mock_tensor]
        mock_gguf.GGUFReader.return_value = mock_instance

        with patch.dict("sys.modules", {"gguf": mock_gguf}):
            wrapper = GGUFReaderWrapper(Path("fake.gguf"))
            self.assertIn("blk.0.attn_q.weight", wrapper.tensor_names)
            self.assertEqual(wrapper.get_tensor_type("blk.0.attn_q.weight"), 0)
            self.assertIsNone(wrapper.get_tensor_type("nonexistent"))

if __name__ == "__main__":
    unittest.main()
