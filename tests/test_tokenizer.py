import unittest
from polymerge.tokenizer import VOCAB_TENSOR_NAMES

class TestPolyMergeTokenizer(unittest.TestCase):
    def test_vocab_tensor_names(self):
        self.assertIn("model.embed_tokens.weight", VOCAB_TENSOR_NAMES)
        self.assertIn("lm_head.weight", VOCAB_TENSOR_NAMES)
        self.assertIn("transformer.wte.weight", VOCAB_TENSOR_NAMES)

if __name__ == "__main__":
    unittest.main()
