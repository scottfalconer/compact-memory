from __future__ import annotations

"""Experimental engine demonstrating rationale episodes."""

from typing import List, Union, Any

from CompressionEngine.core.registry import register_compression_engine # Updated import
from CompressionEngine.core.engines_abc import ( # Updated import
    CompressionEngine, # Updated class name
    CompressedMemory,
    CompressionTrace,
)


class RationaleEpisodeEngine(CompressionEngine): # Updated class name and inheritance
    """Append a rationale note to each chunk of text."""

    id = "rationale_episode_engine" # Updated id

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
            engine_name=self.id, # Updated parameter name
            engine_params={"llm_token_budget": llm_token_budget}, # Updated parameter name
            input_summary={"input_length": len(text)},
            output_summary={"output_length": len(truncated)},
            final_compressed_object_preview=truncated[:50],
        )
        return compressed, trace

    def save_learnable_components(
        self, path: str
    ) -> None:  # pragma: no cover - no learnables
        pass


register_compression_engine( # Updated function name
    RationaleEpisodeEngine.id, RationaleEpisodeEngine, source="contrib" # Updated class name
)

__all__ = ["RationaleEpisodeEngine"] # Updated class name
