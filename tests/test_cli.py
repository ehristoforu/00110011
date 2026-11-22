import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

from polymerge.cli import print_custom_help, main

class TestCliComprehensive(unittest.TestCase):
    def test_help_execution(self):
        try:
            print_custom_help()
        except Exception as e:
            self.fail(f"print_custom_help failed with: {e}")

    @patch("sys.argv", ["polymerge", "--help"])
    def test_cli_help_flag(self):
        with self.assertRaises(SystemExit) as cm:
            main()
        self.assertEqual(cm.exception.code, 0)

    @patch("sys.argv", ["polymerge", "help"])
    def test_cli_help_command(self):
        with self.assertRaises(SystemExit) as cm:
            main()
        self.assertEqual(cm.exception.code, 0)

    @patch("sys.argv", ["polymerge", "-h"])
    def test_cli_short_h_flag(self):
        with self.assertRaises(SystemExit) as cm:
            main()
        self.assertEqual(cm.exception.code, 0)

    @patch("sys.argv", ["polymerge"])
    def test_cli_no_args_shows_help(self):
        with self.assertRaises(SystemExit) as cm:
            main()
        self.assertIn("Missing required parameter", str(cm.exception.code))

    @patch("sys.argv", ["polymerge", "--audit", "--base", "fake/path"])
    @patch("polymerge.audit.SanityChecker.audit_model")
    def test_cli_audit_dispatch(self, mock_audit):
        mock_audit.return_value = {"passed": True, "nan_inf": 0, "condition_numbers": {}}
        try:
            main()
        except SystemExit:
            pass
        self.assertTrue(mock_audit.called)

    @patch("sys.argv", ["polymerge", "--extract-lora", "--base", "fake/base", "--tuned", "fake/tuned", "-o", "fake/out"])
    @patch("polymerge.lora.LoRATool.extract_lora")
    def test_cli_extract_lora_dispatch(self, mock_extract):
        try:
            main()
        except SystemExit:
            pass
        self.assertTrue(mock_extract.called)

    @patch("sys.argv", ["polymerge", "--merge-lora", "--base", "fake/base", "--adapter", "fake/adapter", "-o", "fake/out"])
    @patch("polymerge.lora.LoRATool.merge_lora")
    def test_cli_merge_lora_dispatch(self, mock_merge):
        try:
            main()
        except SystemExit:
            pass
        self.assertTrue(mock_merge.called)

    @patch("sys.argv", ["polymerge", "--merge-adapters", "--adapters", "a1", "a2", "-o", "fake/out"])
    @patch("polymerge.lora.LoRATool.merge_adapters")
    def test_cli_merge_adapters_dispatch(self, mock_merge_ad):
        try:
            main()
        except SystemExit:
            pass
        self.assertTrue(mock_merge_ad.called)

    @patch("sys.argv", ["polymerge", "--prune-dense", "--base", "fake/base", "-o", "fake/out"])
    @patch("polymerge.prune.ModelPruner.prune_dense_layers")
    def test_cli_prune_dense_dispatch(self, mock_prune):
        try:
            main()
        except SystemExit:
            pass
        self.assertTrue(mock_prune.called)

    @patch("sys.argv", ["polymerge", "--prune-moe", "--base", "fake/base", "-o", "fake/out", "--target-experts", "4"])
    @patch("polymerge.prune.ModelPruner.prune_moe_experts")
    def test_cli_prune_moe_dispatch(self, mock_prune_moe):
        try:
            main()
        except SystemExit:
            pass
        self.assertTrue(mock_prune_moe.called)

    @patch("sys.argv", ["polymerge", "--align-vocab", "--base", "fake/base", "--models", "m1", "m2", "-o", "fake/out"])
    @patch("polymerge.tokenizer.TokenizerTool.align_vocabularies")
    def test_cli_align_vocab_dispatch(self, mock_align):
        try:
            main()
        except SystemExit:
            pass
        self.assertTrue(mock_align.called)

    @patch("sys.argv", ["polymerge", "--truncate-vocab", "32000", "--base", "fake/base", "-o", "fake/out"])
    @patch("polymerge.tokenizer.TokenizerTool.truncate_vocab")
    def test_cli_truncate_vocab_dispatch(self, mock_trunc):
        try:
            main()
        except SystemExit:
            pass
        self.assertTrue(mock_trunc.called)

if __name__ == "__main__":
    unittest.main()
