from __future__ import annotations

"""Pipeline compression engine for chaining multiple engines."""

from dataclasses import dataclass, field
from typing import List, Union, Any

from . import BaseCompressionEngine, CompressedMemory, CompressionTrace
from .registry import get_compression_engine


@dataclass
class StrategyConfig:
    """Configuration for creating a :class:`BaseCompressionEngine`."""

    strategy_name: str
    strategy_params: dict[str, Any] = field(default_factory=dict)

    def create(self) -> BaseCompressionEngine:
        cls = get_compression_engine(self.strategy_name)
        return cls(**self.strategy_params)


@dataclass
class PipelineEngineConfig:
    """Configuration for :class:`PipelineEngine`."""

    strategies: List[StrategyConfig] = field(default_factory=list)

    def create(self) -> "PipelineEngine":
        return PipelineEngine(self)


class PipelineEngine(BaseCompressionEngine):
    """Execute a sequence of other compression engines."""

    id = "pipeline"

    def __init__(
        self,
        config_or_strategies: Union[PipelineEngineConfig, List[BaseCompressionEngine]],
    ) -> None:
        super().__init__()
        if isinstance(config_or_strategies, PipelineEngineConfig):
            self.config = config_or_strategies
            self.strategies = [
                get_compression_engine(cfg.strategy_name)(**cfg.strategy_params)
                for cfg in self.config.strategies
            ]
        else:
            self.config = PipelineEngineConfig()
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
            engine_name=self.id,
            strategy_params={},
            input_summary={"num_steps": len(self.strategies)},
            steps=step_traces,
            output_summary={"output_length": len(final.text)},
            final_compressed_object_preview=final.text[:50],
        )
        return final, pipeline_trace


__all__ = ["StrategyConfig", "PipelineEngineConfig", "PipelineEngine"]
