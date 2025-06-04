from __future__ import annotations

"""Experimental strategy chaining built-in steps."""

from typing import List, Union, Any, Optional # Added Optional
from compact_memory.chunking import ChunkFn # Added ChunkFn

from compact_memory.compression import NoCompression, ImportanceCompression
from compact_memory.compression.pipeline_strategy import PipelineCompressionStrategy
from compact_memory.compression.strategies_abc import (
    CompressionStrategy,
    CompressedMemory,
    CompressionTrace,
)


class ChainedStrategy(CompressionStrategy):
    """Simple pipeline of importance filtering followed by truncation."""

    id = "chained"

    def __init__(self) -> None:
        self.pipeline = PipelineCompressionStrategy(
            [ImportanceCompression(), NoCompression()]
        )

    def compress(
        self,
        text: str, # Changed
        llm_token_budget: int,
        chunk_fn: Optional[ChunkFn] = None, # Added
        **kwargs: Any,
    ) -> tuple[CompressedMemory, CompressionTrace]:
        return self.pipeline.compress(text, llm_token_budget, chunk_fn=chunk_fn, **kwargs)

    def save_learnable_components(
        self, path: str
    ) -> None:  # pragma: no cover - no learnables
        pass
