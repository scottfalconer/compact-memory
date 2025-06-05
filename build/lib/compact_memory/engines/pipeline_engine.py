from __future__ import annotations

"""Pipeline compression engine for chaining multiple engines."""

from dataclasses import dataclass, field
from typing import List, Union, Any

from . import BaseCompressionEngine, CompressedMemory, CompressionTrace
from .registry import get_compression_engine


@dataclass
class EngineConfig:
    """Configuration for creating a :class:`BaseCompressionEngine`."""

    engine_name: str
    engine_params: dict[str, Any] = field(default_factory=dict)

    def create(self) -> BaseCompressionEngine:
        cls = get_compression_engine(self.engine_name)
        return cls(**self.engine_params)


@dataclass
class PipelineConfig:
    """Configuration for :class:`PipelineEngine`."""

    engines: List[EngineConfig] = field(default_factory=list)

    def create(self) -> "PipelineEngine":
        return PipelineEngine(self)


class PipelineEngine(BaseCompressionEngine):
    """Execute a sequence of other compression engines."""

    id = "pipeline"

    def __init__(
        self,
        config_or_engines: Union[PipelineConfig, List[BaseCompressionEngine]],
    ) -> None:
        super().__init__()
        if isinstance(config_or_engines, PipelineConfig):
            self.config = config_or_engines
            self.engines = [
                get_compression_engine(cfg.engine_name)(**cfg.engine_params)
                for cfg in self.config.engines
            ]
        else:
            self.config = PipelineConfig()
            self.engines = list(config_or_engines)

    def compress(
        self,
        text_or_chunks: Union[str, List[str]],
        llm_token_budget: int,
        **kwargs: Any,
    ) -> tuple[CompressedMemory, CompressionTrace]:
        current = text_or_chunks
        step_traces = []
        for engine in self.engines:
            compressed, trace = engine.compress(current, llm_token_budget, **kwargs)
            step_traces.append({"strategy": engine.id, "trace": trace})
            current = compressed.text
        final = CompressedMemory(text=current)
        pipeline_trace = CompressionTrace(
            engine_name=self.id,
            strategy_params={},
            input_summary={"num_steps": len(self.engines)},
            steps=step_traces,
            output_summary={"output_length": len(final.text)},
            final_compressed_object_preview=final.text[:50],
        )
        return final, pipeline_trace


__all__ = ["EngineConfig", "PipelineConfig", "PipelineEngine"]
