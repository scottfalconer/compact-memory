from __future__ import annotations

"""Pipeline compression strategy for chaining multiple strategies."""

from dataclasses import dataclass, field
from typing import List, Union, Any, Optional # Added Optional
from compact_memory.chunking import ChunkFn # Added ChunkFn

from .strategies_abc import (
    CompressionStrategy,
    CompressedMemory,
    CompressionTrace,
)
from .config import StrategyConfig


@dataclass
class PipelineStrategyConfig:
    """Configuration for :class:`PipelineCompressionStrategy`."""

    strategies: List[StrategyConfig] = field(default_factory=list)

    def create(self) -> "PipelineCompressionStrategy":
        return PipelineCompressionStrategy(self)


class PipelineCompressionStrategy(CompressionStrategy):
    """Execute a sequence of other compression strategies."""

    id = "pipeline"

    def __init__(self, config_or_strategies: Union[PipelineStrategyConfig, List[CompressionStrategy]]):
        if isinstance(config_or_strategies, PipelineStrategyConfig):
            self.config = config_or_strategies
            self.strategies = [cfg.create() for cfg in self.config.strategies]
        else:
            self.config = PipelineStrategyConfig()
            self.strategies = list(config_or_strategies)

    def compress(
        self,
        text: str, # Changed
        llm_token_budget: int,
        chunk_fn: Optional[ChunkFn] = None, # Added
        **kwargs: Any,
    ) -> tuple[CompressedMemory, CompressionTrace]:
        current_text = text # Initial input text
        step_traces = []

        # The original text_or_chunks could be List[str].
        # The new 'text' is str. If the first strategy in a pre-existing pipeline
        # absolutely needs List[str] and not str + chunk_fn, this could be an issue.
        # However, all strategies are being updated to text: str, chunk_fn: Optional[ChunkFn].
        # So, this should be fine.

        for strat in self.strategies:
            # Each strategy receives the current_text (output of previous, or original text for first)
            # and the original chunk_fn. Each strategy is responsible for deciding how to use chunk_fn.
            compressed, trace = strat.compress(current_text, llm_token_budget, chunk_fn=chunk_fn, **kwargs)
            step_traces.append({"strategy": strat.id, "trace": trace})
            current_text = compressed.text # Output text of one strategy is input to next

        final_compressed_memory = CompressedMemory(text=current_text)
        pipeline_trace = CompressionTrace(
            strategy_name=self.id,
            strategy_params={"num_strategies": len(self.strategies)}, # Added info
            input_summary={"initial_text_char_len": len(text)}, # Trace initial text length
            steps=step_traces,
            output_summary={"final_text_char_len": len(final_compressed_memory.text)}, # Use actual final text length
            final_compressed_object_preview=final_compressed_memory.text[:50],
        )
        return final_compressed_memory, pipeline_trace


__all__ = ["PipelineStrategyConfig", "PipelineCompressionStrategy"]
