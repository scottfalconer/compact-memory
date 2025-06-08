from __future__ import annotations

"""Embedding-based validation metrics."""

from typing import Any, Dict, Optional, List

import numpy as np
import logging

from .. import token_utils

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


class MultiModelEmbeddingSimilarityMetric(ValidationMetric):
    """Compare text similarity across multiple embedding models."""

    metric_id = "multi_model_embedding_similarity"

    def __init__(self, model_names: Optional[List[str]] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model_names = model_names or ["sentence-transformers/all-MiniLM-L6-v2"]

    def _get_tokenizer(self, model_name: str):
        if model_name.startswith("openai/"):
            base_name = model_name.split("/", 1)[1]
            try:
                import tiktoken

                tok = tiktoken.encoding_for_model(base_name)
            except Exception:
                import tiktoken

                tok = tiktoken.get_encoding("gpt2")
            max_len = getattr(tok, "n_ctx", None)
            setattr(tok, "model_max_length", max_len)
            return tok
        try:
            from transformers import AutoTokenizer
        except Exception:
            from ..local_llm import AutoTokenizer  # pragma: no cover - fallback

        return AutoTokenizer.from_pretrained(model_name)

    def evaluate(
        self,
        *,
        original_text: Optional[str] = None,
        compressed_text: Optional[str] = None,
        llm_response: Optional[str] = None,
        reference_answer: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        if original_text is not None and compressed_text is not None:
            text_a, text_b = original_text, compressed_text
        elif llm_response is not None and reference_answer is not None:
            text_a, text_b = reference_answer, llm_response
        else:
            raise ValueError(
                "MultiModelEmbeddingSimilarityMetric requires original/compressed texts or response/reference texts."
            )

        if not text_a or not text_b:
            return {"embedding_similarity": {}}

        results: Dict[str, Dict[str, float]] = {}
        for name in self.model_names:
            try:
                tokenizer = self._get_tokenizer(name)
            except Exception as exc:  # pragma: no cover - tokenizer load failure
                logging.warning("Failed loading tokenizer for %s: %s", name, exc)
                continue

            max_len = getattr(tokenizer, "model_max_length", None)
            if isinstance(max_len, int) and max_len > 0:
                len_a = token_utils.token_count(tokenizer, text_a)
                len_b = token_utils.token_count(tokenizer, text_b)
                if len_a > max_len or len_b > max_len:
                    logging.warning(
                        "Input exceeds model_max_length for %s; skipping", name
                    )
                    continue

            vecs = ep.embed_text([text_a, text_b], model_name=name)
            similarity = float(np.dot(vecs[0], vecs[1]))
            token_count_b = token_utils.token_count(tokenizer, text_b)
            results[name] = {
                "token_count": token_count_b,
                "similarity": similarity,
            }

        return {"embedding_similarity": results}


register_validation_metric(
    MultiModelEmbeddingSimilarityMetric.metric_id, MultiModelEmbeddingSimilarityMetric
)

__all__ = ["EmbeddingSimilarityMetric", "MultiModelEmbeddingSimilarityMetric"]
