from __future__ import annotations

"""Pipeline compression strategy for chaining multiple strategies."""

from dataclasses import dataclass, field
from typing import List, Union, Any

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

    def __init__(
        self,
        config_or_strategies: Union[PipelineStrategyConfig, List[CompressionStrategy]],
    ):
        if isinstance(config_or_strategies, PipelineStrategyConfig):
            self.config = config_or_strategies
            self.strategies = [cfg.create() for cfg in self.config.strategies]
        else:
            self.config = PipelineStrategyConfig()
            self.strategies = list(config_or_strategies)

    def compress(
        self,
        text_or_chunks: Union[str, List[str]],
        llm_token_budget: int,
        **kwargs: Any,
    ) -> tuple[CompressedMemory, CompressionTrace]:
        current = text_or_chunks
        step_traces = []
        for strat in self.strategies:
            compressed, trace = strat.compress(current, llm_token_budget, **kwargs)
            step_traces.append({"strategy": strat.id, "trace": trace})
            current = compressed.text
        final = CompressedMemory(text=current)
        pipeline_trace = CompressionTrace(
            strategy_name=self.id,
            strategy_params={},
            input_summary={"num_steps": len(self.strategies)},
            steps=step_traces,
            output_summary={"output_length": len(final.text)},
            final_compressed_object_preview=final.text[:50],
        )
        return final, pipeline_trace


__all__ = ["PipelineStrategyConfig", "PipelineCompressionStrategy"]
