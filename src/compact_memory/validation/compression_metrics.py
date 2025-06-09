from __future__ import annotations

"""Metrics for evaluating compression quality."""

from typing import Dict

# TODO: Move DEFAULT_TOKENIZER_NAME and get_tokenizer to compact_memory.model_utils
# if they are intended to be globally available, or define them based on project needs.
from transformers import AutoTokenizer

from compact_memory.token_utils import token_count, tokenize_text

from .metrics_abc import ValidationMetric
from .registry import register_validation_metric


class CompressionRatioMetric(ValidationMetric):
    """Ratio of compressed text length to original text length, character and token based."""

    metric_id = "compression_ratio"
    DEFAULT_TOKENIZER_NAME = "gpt2"  # Using a common small tokenizer

    def _get_tokenizer(self, tokenizer_name: str):
        """Helper function to get a tokenizer."""
        return AutoTokenizer.from_pretrained(tokenizer_name)

    def evaluate(self, *, original_text: str, compressed_text: str, **kwargs) -> Dict[str, float]:  # type: ignore[override]
        if not original_text:
            return {"char_compression_ratio": 0.0, "token_compression_ratio": 0.0}

        char_ratio = len(compressed_text) / len(original_text)

        tokenizer = self._get_tokenizer(self.DEFAULT_TOKENIZER_NAME)

        original_token_count = token_count(tokenizer, original_text)
        compressed_token_count = token_count(tokenizer, compressed_text)

        if original_token_count == 0:
            token_ratio = 0.0
        else:
            token_ratio = compressed_token_count / original_token_count

        return {"char_compression_ratio": char_ratio, "token_compression_ratio": token_ratio}


register_validation_metric(CompressionRatioMetric.metric_id, CompressionRatioMetric)

__all__ = ["CompressionRatioMetric"]
