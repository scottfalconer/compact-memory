from __future__ import annotations

"""Pipeline compression engine for chaining multiple engines."""

from dataclasses import dataclass, field
from typing import List, Union, Any

from .engines_abc import (
    CompressionEngine,
    CompressedMemory,
    CompressionTrace,
)
from .config import EngineConfig


@dataclass
class PipelineEngineConfig:
    """Configuration for :class:`PipelineCompressionEngine`."""

    engines: List[EngineConfig] = field(default_factory=list)

    def create(self) -> "PipelineCompressionEngine":
        return PipelineCompressionEngine(self)


class PipelineCompressionEngine(CompressionEngine):
    """Execute a sequence of other compression engines."""

    id = "pipeline_engine"

    def __init__(
        self,
        config_or_engines: Union[PipelineEngineConfig, List[CompressionEngine]],
    ):
        if isinstance(config_or_engines, PipelineEngineConfig):
            self.config = config_or_engines
            self.engines = [cfg.create() for cfg in self.config.engines]
        else:
            self.config = PipelineEngineConfig()
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
            step_traces.append({"engine": engine.id, "trace": trace})
            current = compressed.text
        final = CompressedMemory(text=current)
        pipeline_trace = CompressionTrace(
            engine_name=self.id,
            engine_params={},
            input_summary={"num_steps": len(self.engines)},
            steps=step_traces,
            output_summary={"output_length": len(final.text)},
            final_compressed_object_preview=final.text[:50],
        )
        return final, pipeline_trace


__all__ = ["PipelineEngineConfig", "PipelineCompressionEngine"]
