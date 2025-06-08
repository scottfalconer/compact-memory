from __future__ import annotations

"""Embedding-based validation metrics."""

from typing import Any, Dict, Optional, Sequence

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


class MultiEmbeddingSimilarityMetric(ValidationMetric):
    """Cosine similarity using multiple embedding models."""

    metric_id = "embedding_similarity_multi"

    def __init__(
        self,
        model_names: Sequence[str] | None = None,
        max_tokens: int = 8192,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_names = list(model_names) if model_names else [ep._MODEL_NAME]
        self.max_tokens = int(max_tokens)

    def _token_count(self, a: str, b: str) -> int:
        return len((a + " " + b).split())

    def _max_allowed_tokens(self) -> int:
        limit = self.max_tokens
        for name in self.model_names:
            try:
                model = ep._load_model(name, self.config_params.get("device", "cpu"))
                ml = getattr(model, "model_max_length", None)
                if isinstance(ml, int):
                    limit = min(limit, ml)
            except Exception:
                continue
        return limit

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
                "MultiEmbeddingSimilarityMetric requires original/compressed texts or response/reference texts."
            )

        tokens = self._token_count(text_a, text_b)
        if tokens > self._max_allowed_tokens():
            return {"token_count": float(tokens)}

        results: Dict[str, float] = {"token_count": float(tokens)}
        scores = []
        embed_kwargs = {}
        for key in ["device", "batch_size"]:
            if key in self.config_params:
                embed_kwargs[key] = self.config_params[key]
        for name in self.model_names:
            vecs = ep.embed_text([text_a, text_b], model_name=name, **embed_kwargs)
            s = float(np.dot(vecs[0], vecs[1]))
            scores.append(s)
            results[name] = s
        results["semantic_similarity"] = float(np.mean(scores)) if scores else 0.0
        return results


register_validation_metric(
    EmbeddingSimilarityMetric.metric_id, EmbeddingSimilarityMetric
)

register_validation_metric(
    MultiEmbeddingSimilarityMetric.metric_id, MultiEmbeddingSimilarityMetric
)

__all__ = ["EmbeddingSimilarityMetric", "MultiEmbeddingSimilarityMetric"]
