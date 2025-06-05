from __future__ import annotations

"""Experimental engine chaining built-in steps."""

from typing import List, Union, Any

from CompressionEngine.core import register_compression_engine, NoCompressionEngine
from CompressionEngine.core.pipeline_strategy import PipelineCompressionEngine
from CompressionEngine.core.engines_abc import (
    CompressionEngine,
    CompressedMemory,
    CompressionTrace,
)


class ChainedEngine(CompressionEngine):
    """Simple pipeline of multiple compression steps."""

    id = "chained_engine"

    def __init__(self) -> None:
        self.pipeline = PipelineCompressionEngine([NoCompressionEngine()])

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


register_compression_engine(ChainedEngine.id, ChainedEngine, source="contrib")

__all__ = ["ChainedEngine"]
