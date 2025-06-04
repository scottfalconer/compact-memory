from __future__ import annotations

"""Experimental strategy demonstrating rationale episodes."""

from typing import List, Union, Any, Optional # Added Optional
from compact_memory.chunking import ChunkFn # Added ChunkFn

from compact_memory.compression.strategies_abc import (
    CompressionStrategy,
    CompressedMemory,
    CompressionTrace,
)


class RationaleEpisodeStrategy(CompressionStrategy):
    """Append a rationale note to each chunk of text."""

    id = "rationale_episode"

    def compress(
        self,
        text: str, # Changed
        llm_token_budget: int,
        chunk_fn: Optional[ChunkFn] = None, # Added
        **kwargs: Any,
    ) -> tuple[CompressedMemory, CompressionTrace]:
        if chunk_fn:
            chunks = chunk_fn(text)
        else:
            chunks = [text]

        processed_text = " ".join(chunks)

        result = f"Rationale: {processed_text}"
        # llm_token_budget is char limit here; if 0 or negative, means no truncation by budget
        # However, ABC defines it as int, not Optional[int]. Standardizing to "keep all if budget <= 0"
        if llm_token_budget > 0:
            truncated_text = result[:llm_token_budget]
        else:
            truncated_text = result # Keep all if budget is not positive

        compressed = CompressedMemory(text=truncated_text)
        trace = CompressionTrace(
            strategy_name=self.id,
            strategy_params={"llm_token_budget": llm_token_budget, "chunked_input": chunk_fn is not None},
            input_summary={"input_length": len(processed_text), "num_chunks": len(chunks)},
            output_summary={"output_length": len(truncated_text)},
            final_compressed_object_preview=truncated_text[:50],
        )
        return compressed, trace

    def save_learnable_components(
        self, path: str
    ) -> None:  # pragma: no cover - no learnables
        pass
