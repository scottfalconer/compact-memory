from __future__ import annotations

"""Metrics for evaluating compression quality."""

from typing import Dict

from .metrics_abc import ValidationMetric
from .registry import register_validation_metric


class CompressionRatioMetric(ValidationMetric):
    """Ratio of compressed text length to original text length."""

    metric_id = "compression_ratio"

    def evaluate(self, *, original_text: str, compressed_text: str, **kwargs) -> Dict[str, float]:  # type: ignore[override]
        if not original_text:
            return {"compression_ratio": 0.0}
        ratio = len(compressed_text) / len(original_text)
        return {"compression_ratio": ratio}


register_validation_metric(CompressionRatioMetric.metric_id, CompressionRatioMetric)

__all__ = ["CompressionRatioMetric"]
