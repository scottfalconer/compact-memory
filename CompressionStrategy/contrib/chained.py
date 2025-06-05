from __future__ import annotations

"""Experimental strategy chaining built-in steps."""

from typing import List, Union, Any

from CompressionStrategy.core import register_compression_strategy, NoCompression
from CompressionStrategy.core.pipeline_strategy import PipelineCompressionStrategy
from CompressionStrategy.core.strategies_abc import (
    CompressionStrategy,
    CompressedMemory,
    CompressionTrace,
)


class ChainedStrategy(CompressionStrategy):
    """Simple pipeline of multiple compression steps."""

    id = "chained"

    def __init__(self) -> None:
        self.pipeline = PipelineCompressionStrategy([NoCompression()])

    def compress(
        self,
        text_or_chunks: Union[str, List[str]],
        llm_token_budget: int,
        **kwargs: Any,
    ) -> tuple[CompressedMemory, CompressionTrace]:
        return self.pipeline.compress(text_or_chunks, llm_token_budget, **kwargs)

    def save_learnable_components(
        self, path: str
    ) -> None:  # pragma: no cover - no learnables
        pass


register_compression_strategy(ChainedStrategy.id, ChainedStrategy, source="contrib")

__all__ = ["ChainedStrategy"]
