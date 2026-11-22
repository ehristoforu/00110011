import unittest
from polymerge.moe import detect_dense_mlp, MoEAssembler
from polymerge.slicer import LayerSlicer
from polymerge.lora import LoRATool
from polymerge.prune import ModelPruner
from polymerge.tokenizer import TokenizerTool, VOCAB_TENSOR_NAMES
from polymerge.utils import human_bytes, select_device, setup_fast_hf_transfer

class TestFullProjectParameters(unittest.TestCase):
    def test_moe_dense_mlp_patterns(self):
        # SwiGLU 3-projection patterns
        self.assertEqual(detect_dense_mlp("model.layers.0.mlp.gate_proj.weight")[1], "w1")
        self.assertEqual(detect_dense_mlp("model.layers.0.mlp.down_proj.weight")[1], "w2")
        self.assertEqual(detect_dense_mlp("model.layers.0.mlp.up_proj.weight")[1], "w3")
        self.assertEqual(detect_dense_mlp("model.layers.0.mlp.w1.weight")[1], "w1")
        self.assertEqual(detect_dense_mlp("model.layers.0.mlp.w2.weight")[1], "w2")
        self.assertEqual(detect_dense_mlp("model.layers.0.mlp.w3.weight")[1], "w3")

        # 2-projection patterns
        self.assertEqual(detect_dense_mlp("model.layers.0.mlp.fc1.weight")[1], "w1")
        self.assertEqual(detect_dense_mlp("model.layers.0.mlp.fc2.weight")[1], "w2")
        self.assertEqual(detect_dense_mlp("model.layers.0.mlp.c_fc.weight")[1], "w1")
        self.assertEqual(detect_dense_mlp("model.layers.0.mlp.c_proj.weight")[1], "w2")

        # Non-MLP patterns
        self.assertIsNone(detect_dense_mlp("model.layers.0.self_attn.q_proj.weight"))
        self.assertIsNone(detect_dense_mlp("model.layers.0.input_layernorm.weight"))

    def test_slicer_validation(self):
        slicer = LayerSlicer(
            slices=[
                {"sources": [{"model": "model_1", "layer_range": [0, 16]}]},
                {"sources": [{"model": "model_2", "layer_range": [8, 24]}]},
            ],
            chunk_size_bytes=1_000_000_000,
        )
        self.assertEqual(len(slicer.slices), 2)
        self.assertEqual(slicer.chunk_size_bytes, 1_000_000_000)

    def test_pruner_module_methods(self):
        self.assertTrue(callable(getattr(ModelPruner, "prune_dense_layers")))
        self.assertTrue(callable(getattr(ModelPruner, "prune_moe_experts")))

    def test_tokenizer_methods(self):
        self.assertTrue(callable(getattr(TokenizerTool, "align_vocabularies")))
        self.assertTrue(callable(getattr(TokenizerTool, "truncate_vocab")))
        self.assertGreater(len(VOCAB_TENSOR_NAMES), 2)

    def test_lora_methods(self):
        self.assertTrue(callable(getattr(LoRATool, "extract_lora")))
        self.assertTrue(callable(getattr(LoRATool, "merge_lora")))
        self.assertTrue(callable(getattr(LoRATool, "merge_adapters")))

    def test_hardware_utils(self):
        setup_fast_hf_transfer()
        self.assertEqual(human_bytes(512), "512.0 B")
        self.assertEqual(human_bytes(1024 * 1024), "1.0 MB")
        self.assertEqual(human_bytes(1024 * 1024 * 1024 * 5), "5.0 GB")

        # CPU device selection always valid even without CUDA/TPU
        dev = select_device("cpu")
        self.assertEqual(dev.type, "cpu")

if __name__ == "__main__":
    unittest.main()
