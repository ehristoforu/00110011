import unittest
from polymerge.moe import detect_dense_mlp

class TestPolyMergeMoE(unittest.TestCase):
    def test_detect_dense_mlp(self):
        res1 = detect_dense_mlp("model.layers.0.mlp.gate_proj.weight")
        self.assertIsNotNone(res1)
        self.assertEqual(res1[0], "model.layers.0.mlp")
        self.assertEqual(res1[1], "w1")

        res2 = detect_dense_mlp("model.layers.12.mlp.down_proj.weight")
        self.assertIsNotNone(res2)
        self.assertEqual(res2[0], "model.layers.12.mlp")
        self.assertEqual(res2[1], "w2")

        res3 = detect_dense_mlp("model.layers.5.mlp.up_proj.weight")
        self.assertIsNotNone(res3)
        self.assertEqual(res3[0], "model.layers.5.mlp")
        self.assertEqual(res3[1], "w3")

        res4 = detect_dense_mlp("model.layers.0.self_attn.q_proj.weight")
        self.assertIsNone(res4)

        res5 = detect_dense_mlp("model.embed_tokens.weight")
        self.assertIsNone(res5)

if __name__ == "__main__":
    unittest.main()
