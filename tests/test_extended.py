import unittest
from polymerge.methods import get_merge_method
from polymerge.audit import SanityChecker
from polymerge.publisher import PublisherTool
from polymerge.quantizer import QuantizerTool
from polymerge.evolve import EvolutionOptimizer

class TestExtendedFeatures(unittest.TestCase):
    def test_new_merge_methods_registered(self):
        self.assertIsNotNone(get_merge_method("karcher"))
        self.assertIsNotNone(get_merge_method("nuslerp"))
        self.assertIsNotNone(get_merge_method("nearswap"))

    def test_audit_module_callable(self):
        self.assertTrue(callable(getattr(SanityChecker, "audit_model")))

    def test_publisher_module_callable(self):
        self.assertTrue(callable(getattr(PublisherTool, "push_to_hub")))

    def test_quantizer_module_callable(self):
        self.assertTrue(callable(getattr(QuantizerTool, "quantize_gguf")))

    def test_evolution_module_callable(self):
        self.assertTrue(callable(getattr(EvolutionOptimizer, "optimize_weights_grid")))

if __name__ == "__main__":
    unittest.main()
