from __future__ import annotations

"""Example CompressionStrategy using a pretrained summarization model."""

from typing import Any, Dict, List, Union, Tuple

from CompressionStrategy.core.strategies_abc import (
    CompressedMemory,
    CompressionStrategy,
    CompressionTrace,
)


class LearnedSummarizerStrategy(CompressionStrategy):
    """Proof-of-concept strategy leveraging a summarization model."""

    id = "learned_summarizer"

    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-12-6") -> None:
        self.model_name = model_name
        try:
            from transformers import pipeline
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "transformers with PyTorch is required for LearnedSummarizerStrategy"
            ) from exc
        self.summarizer = pipeline("summarization", model=model_name)

    def compress(
        self,
        text_or_chunks: Union[str, List[str]],
        llm_token_budget: int,
        **kwargs: Any,
    ) -> Tuple[CompressedMemory, CompressionTrace]:
        if isinstance(text_or_chunks, list):
            text = " ".join(text_or_chunks)
        else:
            text = str(text_or_chunks)
        result = self.summarizer(
            text, max_length=llm_token_budget, min_length=1, do_sample=False
        )
        summary = result[0]["summary_text"]
        compressed = CompressedMemory(text=summary)
        trace = CompressionTrace(
            strategy_name=self.id,
            strategy_params={"model_name": self.model_name, "budget": llm_token_budget},
            input_summary={"chars": len(text)},
            output_summary={"chars": len(summary)},
            processing_ms=None,
            final_compressed_object_preview=summary[:50],
        )
        return compressed, trace

    def save_learnable_components(self, path: str) -> None:
        self.summarizer.save_pretrained(path)  # type: ignore[attr-defined]

    def load_learnable_components(self, path: str) -> None:
        try:
            from transformers import pipeline
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "transformers with PyTorch is required for LearnedSummarizerStrategy"
            ) from exc
        self.summarizer = pipeline("summarization", model=path)
