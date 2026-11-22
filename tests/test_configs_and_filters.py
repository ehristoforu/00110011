import unittest
import glob
from pathlib import Path
from polymerge.config import MergeConfig
from polymerge.filter import TensorFilter

class TestAllConfigExamples(unittest.TestCase):
    def test_all_yaml_configs_parseable(self):
        yaml_files = glob.glob("examples_configs/*.yaml")
        self.assertGreater(len(yaml_files), 5)
        for yf in yaml_files:
            try:
                cfg = MergeConfig.from_yaml(yf)
                self.assertIsNotNone(cfg)
                self.assertIsInstance(cfg.models, list)
            except Exception as e:
                self.fail(f"Failed to parse example config {yf}: {e}")

class TestFilterExhaustive(unittest.TestCase):
    def setUp(self):
        self.f = TensorFilter(ignore_vision=True)

    def test_vision_architectures(self):
        vision_samples = [
            "model.vision_tower.encoder.layers.0.self_attn.k_proj.weight",
            "visual.blocks.12.mlp.fc1.weight",
            "model.vlm.vision_projection.weight",
            "model.image_newline",
            "multi_modal_projector.linear_2.bias",
            "model.mm_projector.0.weight",
            "clip.visual.conv1.weight",
            "vit.encoder.layer.0.attention.weight",
        ]
        for name in vision_samples:
            self.assertTrue(self.f.should_ignore(name), f"Should ignore vision layer: {name}")

    def test_text_architectures_preserved(self):
        text_samples = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.input_layernorm.weight",
            "model.norm.weight",
            "lm_head.weight",
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.0.mlp.shared_expert.down_proj.weight",
            "model.layers.0.mlp.gate.weight",
            "model.layers.0.mlp.router.proj.weight",
            "model.layers.0.mlp.router.scale",
            "model.layers.0.mlp.router.per_expert_scale",
        ]
        for name in text_samples:
            self.assertFalse(self.f.should_ignore(name), f"Should NOT ignore text layer: {name}")

    def test_custom_skip_patterns(self):
        f = TensorFilter(skip_patterns=["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"])
        self.assertTrue(f.should_skip_merge("model.layers.0.mlp.gate_proj.weight"))
        self.assertTrue(f.should_skip_merge("model.layers.5.mlp.down_proj.weight"))
        self.assertFalse(f.should_skip_merge("model.layers.0.self_attn.q_proj.weight"))

if __name__ == "__main__":
    unittest.main()
