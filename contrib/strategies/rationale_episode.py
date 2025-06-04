from __future__ import annotations

"""Experimental strategy demonstrating rationale episodes."""

from typing import List, Union, Any

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
        text_or_chunks: Union[str, List[str]],
        llm_token_budget: int,
        **kwargs: Any,
    ) -> tuple[CompressedMemory, CompressionTrace]:
        if isinstance(text_or_chunks, list):
            text = " ".join(text_or_chunks)
        else:
            text = text_or_chunks
        result = f"Rationale: {text}"
        truncated = result[:llm_token_budget] if llm_token_budget else result
        compressed = CompressedMemory(text=truncated)
        trace = CompressionTrace(
            strategy_name=self.id,
            strategy_params={"llm_token_budget": llm_token_budget},
            input_summary={"input_length": len(text)},
            output_summary={"output_length": len(truncated)},
            final_compressed_object_preview=truncated[:50],
        )
        return compressed, trace

    def save_learnable_components(
        self, path: str
    ) -> None:  # pragma: no cover - no learnables
        pass
