from __future__ import annotations

import unittest

from compact_memory.validation.compression_metrics import CompressionRatioMetric


class TestCompressionRatioMetric(unittest.TestCase):
    def test_evaluate_dual_ratios(self):
        metric = CompressionRatioMetric()

        # Test case 1: Regular text
        original_text = "This is a sample text for testing."
        compressed_text = "Ths smpl txt 4 tstng."
        # Expected char ratio: len("Ths smpl txt 4 tstng.") / len("This is a sample text for testing.")
        # Expected char ratio: 22 / 34 = 0.6470588235
        # Expected token ratio depends on the tokenizer, let's assume simple space tokenization for estimation
        # "This is a sample text for testing." -> 7 tokens
        # "Ths smpl txt 4 tstng." -> 5 tokens
        # Expected token ratio: 5 / 7 = 0.714285714
        # Note: Actual tokenization will be different with a real tokenizer.
        # We will assert that the values are calculated, actual values depend on tokenizer.
        result1 = metric.evaluate(original_text=original_text, compressed_text=compressed_text)
        self.assertIn("char_compression_ratio", result1)
        self.assertIn("token_compression_ratio", result1)
        self.assertAlmostEqual(result1["char_compression_ratio"], 21 / 34, places=7)
        # We cannot assert exact token ratio without knowing the tokenizer's behavior.
        # Instead, we check if it's a float and non-negative.
        self.assertIsInstance(result1["token_compression_ratio"], float)
        self.assertGreaterEqual(result1["token_compression_ratio"], 0.0)

        # Test case 2: Empty original text
        original_text_empty = ""
        compressed_text_empty = ""
        result2 = metric.evaluate(original_text=original_text_empty, compressed_text=compressed_text_empty)
        self.assertEqual(result2["char_compression_ratio"], 0.0)
        self.assertEqual(result2["token_compression_ratio"], 0.0)

        # Test case 3: Text where tokenization might differ significantly
        # Repeating characters might be single token or multiple, depending on tokenizer.
        original_text_tokens = "aaaaa bbbbb ccccc" # Expected 3 tokens with space tokenizer
        compressed_text_tokens = "a b c" # Expected 3 tokens
        # Char ratio: 5 / 17 = 0.294117647
        # Token ratio could be 3/3 = 1.0 if "aaaaa" is one token, or different otherwise.
        result3 = metric.evaluate(original_text=original_text_tokens, compressed_text=compressed_text_tokens)
        self.assertIn("char_compression_ratio", result3)
        self.assertIn("token_compression_ratio", result3)
        self.assertAlmostEqual(result3["char_compression_ratio"], 5 / 17, places=7)
        self.assertIsInstance(result3["token_compression_ratio"], float)
        self.assertGreaterEqual(result3["token_compression_ratio"], 0.0)

        # Test case 4: Compressed text is longer than original (char length)
        original_text_short = "test"
        compressed_text_long = "this is a longer compressed text"
        result4 = metric.evaluate(original_text=original_text_short, compressed_text=compressed_text_long)
        self.assertAlmostEqual(result4["char_compression_ratio"], len(compressed_text_long) / len(original_text_short), places=7)
        self.assertIsInstance(result4["token_compression_ratio"], float)
        self.assertGreaterEqual(result4["token_compression_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()
