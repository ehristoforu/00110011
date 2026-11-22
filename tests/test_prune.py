import unittest
from polymerge.prune import ModelPruner

class TestPolyMergePruner(unittest.TestCase):
    def test_pruner_module_exists(self):
        self.assertTrue(hasattr(ModelPruner, "prune_dense_layers"))
        self.assertTrue(hasattr(ModelPruner, "prune_moe_experts"))

if __name__ == "__main__":
    unittest.main()
