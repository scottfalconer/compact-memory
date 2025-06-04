from __future__ import annotations

"""Example CompressionStrategy using a pretrained summarization model."""

from typing import Any, Dict, List, Union, Tuple, Optional # Added Optional
from compact_memory.chunking import ChunkFn # Added ChunkFn
from transformers import pipeline

from .compression.strategies_abc import CompressedMemory, CompressionStrategy, CompressionTrace


class LearnedSummarizerStrategy(CompressionStrategy):
    """Proof-of-concept strategy leveraging a summarization model."""

    id = "learned_summarizer"

    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-12-6") -> None:
        self.model_name = model_name
        self.summarizer = pipeline("summarization", model=model_name)

    def compress(
        self,
        text: str, # Changed
        llm_token_budget: int,
        chunk_fn: Optional[ChunkFn] = None, # Added
        **kwargs: Any,
    ) -> Tuple[CompressedMemory, CompressionTrace]:
        if chunk_fn:
            chunks = chunk_fn(text)
        else:
            chunks = [text]

        processed_text = " ".join(chunks)

        # Note: llm_token_budget for summarizer's max_length might mean token count,
        # not char count as previously implied by input_summary's "chars".
        # This strategy should ideally use a tokenizer if budget is in tokens.
        # For now, passing it as max_length.
        result = self.summarizer(processed_text, max_length=llm_token_budget, min_length=1, do_sample=False)
        summary = result[0]["summary_text"]
        compressed = CompressedMemory(text=summary)
        trace = CompressionTrace(
            strategy_name=self.id,
            strategy_params={"model_name": self.model_name, "max_length_budget": llm_token_budget, "chunked_input": chunk_fn is not None},
            input_summary={"chars": len(processed_text), "num_chunks": len(chunks)},
            output_summary={"chars": len(summary)},
            processing_ms=None, # TODO: Consider adding timing for the summarization call
            final_compressed_object_preview=summary[:50],
        )
        return compressed, trace

    def save_learnable_components(self, path: str) -> None:
        self.summarizer.save_pretrained(path)  # type: ignore[attr-defined]

    def load_learnable_components(self, path: str) -> None:
        self.summarizer = pipeline("summarization", model=path)
