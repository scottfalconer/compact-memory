from __future__ import annotations

"""Embedding-based validation metrics."""

from typing import Any, Dict, Optional

import numpy as np

from .. import embedding_pipeline as ep
from .metrics_abc import ValidationMetric
from .registry import register_validation_metric


class EmbeddingSimilarityMetric(ValidationMetric):
    """Cosine similarity between embeddings of two texts."""

    metric_id = "embedding_similarity"

    def evaluate(
        self,
        *,
        original_text: Optional[str] = None,
        compressed_text: Optional[str] = None,
        llm_response: Optional[str] = None,
        reference_answer: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        if original_text is not None and compressed_text is not None:
            text_a, text_b = original_text, compressed_text
        elif llm_response is not None and reference_answer is not None:
            text_a, text_b = reference_answer, llm_response
        else:
            raise ValueError(
                "EmbeddingSimilarityMetric requires original/compressed texts or response/reference texts."
            )

        if not text_a or not text_b:
            return {"semantic_similarity": 0.0}

        embed_kwargs = {}
        for key in ["model_name", "device", "batch_size"]:
            if key in self.config_params:
                embed_kwargs[key] = self.config_params[key]

        vecs = ep.embed_text([text_a, text_b], **embed_kwargs)
        score = float(np.dot(vecs[0], vecs[1]))
        return {"semantic_similarity": score}


register_validation_metric(
    EmbeddingSimilarityMetric.metric_id, EmbeddingSimilarityMetric
)

__all__ = ["EmbeddingSimilarityMetric"]
