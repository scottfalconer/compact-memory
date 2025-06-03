from __future__ import annotations
from typing import List, Union, Any

from .compression import register_compression_strategy
from .compression.strategies_abc import CompressionStrategy, CompressedMemory, CompressionTrace

class FirstLastStrategy(CompressionStrategy):
    """Keep first and last parts of the text within the budget."""

    id = "first_last"

    def compress(
        self,
        text_or_chunks: Union[str, List[str]],
        llm_token_budget: int | None,
        **kwargs: Any,
    ) -> tuple[CompressedMemory, CompressionTrace]:
        text = text_or_chunks if isinstance(text_or_chunks, str) else " ".join(text_or_chunks)
        if llm_token_budget is None or llm_token_budget <= 0:
            kept = text
            half = len(text)
        else:
            half = max(llm_token_budget // 2, 0)
            kept = text[:half] + text[-half:]
        compressed = CompressedMemory(text=kept)
        trace = CompressionTrace(
            strategy_name=self.id,
            strategy_params={"llm_token_budget": llm_token_budget},
            input_summary={"input_length": len(text)},
            steps=[{"type": "first_last", "kept_first": len(kept[:half]), "kept_last": len(kept[half:])}],
            output_summary={"final_length": len(kept)},
            final_compressed_object_preview=kept[:50],
        )
        return compressed, trace


register_compression_strategy(FirstLastStrategy.id, FirstLastStrategy)

__all__ = ["FirstLastStrategy"]
