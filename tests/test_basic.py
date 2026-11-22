import unittest
import os
import tempfile
from pathlib import Path
from polymerge.filter import TensorFilter
from polymerge.config import MergeConfig
from polymerge.utils import human_bytes, setup_fast_hf_transfer

class TestPolyMergeFilter(unittest.TestCase):
    def test_vision_filtering(self):
        f = TensorFilter(ignore_vision=True)
        self.assertTrue(f.should_ignore("model.vision_tower.encoder.layer.0.weight"))
        self.assertTrue(f.should_ignore("visual.blocks.0.attn.qkv.weight"))
        self.assertTrue(f.should_ignore("multi_modal_projector.linear_1.weight"))
        self.assertTrue(f.should_ignore("model.vlm.head.weight"))
        self.assertTrue(f.should_ignore("clip.text_projection"))
        self.assertFalse(f.should_ignore("model.layers.0.self_attn.q_proj.weight"))
        self.assertFalse(f.should_ignore("model.layers.0.mlp.down_proj.weight"))
        self.assertFalse(f.should_ignore("model.embed_tokens.weight"))

    def test_custom_keep_and_ignore(self):
        f = TensorFilter(
            ignore_vision=False,
            custom_ignore_patterns=[r"lm_head\."],
            custom_keep_patterns=[r"vision_tower\.special"],
        )
        self.assertTrue(f.should_ignore("model.lm_head.weight"))
        self.assertFalse(f.should_ignore("model.vision_tower.special.weight"))

    def test_skip_patterns(self):
        f = TensorFilter(skip_patterns=["mlp.gate_proj", "mlp.up_proj"])
        self.assertTrue(f.should_skip_merge("model.layers.0.mlp.gate_proj.weight"))
        self.assertFalse(f.should_skip_merge("model.layers.0.mlp.down_proj.weight"))

class TestPolyMergeConfig(unittest.TestCase):
    def test_config_parsing(self):
        yaml_content = """
        merge_method: dare_ties
        base_model: meta-llama/Llama-3.2-1B
        models:
          - model: meta-llama/Llama-3.2-1B-Instruct
            weight: 0.6
            density: 0.5
        ignore_vision: true
        chunk_size_bytes: 500000000
        """
        cfg = MergeConfig.from_yaml(yaml_content)
        self.assertEqual(cfg.merge_method, "dare_ties")
        self.assertEqual(cfg.base_model, "meta-llama/Llama-3.2-1B")
        self.assertEqual(len(cfg.models), 1)
        self.assertEqual(cfg.models[0].weight, 0.6)
        self.assertEqual(cfg.models[0].density, 0.5)
        self.assertTrue(cfg.ignore_vision)
        self.assertEqual(cfg.chunk_size_bytes, 500000000)

    def test_staged_config_parsing(self):
        yaml_content = """
        staged: true
        stages:
          - name: "Stage 1"
            merge_method: linear
            models:
              - model: "model_a"
                weight: 0.5
              - model: "model_b"
                weight: 0.5
          - name: "Stage 2"
            merge_method: ties
            base_model: "base_model"
            models:
              - model: "PREVIOUS_STAGE"
                weight: 0.8
        """
        cfg = MergeConfig.from_yaml(yaml_content)
        self.assertTrue(cfg.staged)
        self.assertEqual(len(cfg.stages), 2)

class TestPolyMergeUtils(unittest.TestCase):
    def test_human_bytes(self):
        self.assertEqual(human_bytes(1024), "1.0 KB")
        self.assertEqual(human_bytes(1024 * 1024 * 500), "500.0 MB")
        self.assertEqual(human_bytes(1024 * 1024 * 1024 * 2), "2.0 GB")

    def test_setup_fast_hf_transfer(self):
        setup_fast_hf_transfer()
        self.assertEqual(os.environ.get("HF_HUB_ENABLE_HF_TRANSFER"), "1")

if __name__ == "__main__":
    unittest.main()
