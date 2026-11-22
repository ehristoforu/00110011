import unittest
from polymerge.moe import detect_dense_mlp
from polymerge.slicer import LayerSlicer
from polymerge.lora import LoRATool

class TestPolyMergeAdvanced(unittest.TestCase):
    def test_moe_mlp_detection_extended(self):
        # Mixtral / Llama / Qwen style
        res1 = detect_dense_mlp("model.layers.0.mlp.gate_proj.weight")
        self.assertIsNotNone(res1)
        self.assertEqual(res1[1], "w1")

        # Gemma style
        res2 = detect_dense_mlp("model.layers.2.mlp.up_proj.weight")
        self.assertIsNotNone(res2)
        self.assertEqual(res2[1], "w3")

        # Block sparse MoE
        res3 = detect_dense_mlp("model.layers.4.block_sparse_moe.w2.weight")
        self.assertIsNotNone(res3)
        self.assertEqual(res3[1], "w2")

        # 2-layer FFN (GPT-2, Falcon)
        res4 = detect_dense_mlp("transformer.h.0.mlp.c_fc.weight")
        self.assertIsNotNone(res4)
        self.assertEqual(res4[1], "w1")

    def test_slicer_init(self):
        slicer = LayerSlicer([
            {"sources": [{"model": "dummy", "layer_range": [0, 10]}]}
        ])
        self.assertEqual(len(slicer.slices), 1)

if __name__ == "__main__":
    unittest.main()
