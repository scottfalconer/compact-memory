from __future__ import annotations

"""Pipeline compression engine for chaining multiple engines."""

from dataclasses import dataclass, field, asdict
from typing import List, Union, Any, Optional
import time

# Import base classes directly to avoid package-level registration
from .base import BaseCompressionEngine, CompressedMemory, CompressionTrace
from .registry import get_compression_engine, register_compression_engine


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
        text_or_chunks: Union[
            str, List[str]
        ],  # This is the original text for the first engine if no previous_compression_result
        budget: int,  # Renamed from llm_token_budget for consistency
        previous_compression_result: Optional[CompressedMemory] = None,
        **kwargs: Any,
    ) -> CompressedMemory:
        current_compressed_memory: Optional[CompressedMemory] = (
            previous_compression_result
        )
        accumulated_traces: List[CompressionTrace] = []
        start = time.monotonic()

        # Determine the very original text for the pipeline's input summary
        original_input_text = text_or_chunks
        if isinstance(text_or_chunks, List):
            original_input_text = " ".join(text_or_chunks)

        for i, engine in enumerate(self.engines):
            input_text_for_engine: str
            prev_comp_result_for_engine: Optional[CompressedMemory] = None

            if i == 0:  # First engine
                if (
                    current_compressed_memory is None
                ):  # No initial result provided to the pipeline
                    input_text_for_engine = original_input_text
                    prev_comp_result_for_engine = None  # Explicitly
                else:  # Initial result WAS provided to the pipeline
                    input_text_for_engine = current_compressed_memory.text
                    prev_comp_result_for_engine = current_compressed_memory
            else:  # Subsequent engines
                if current_compressed_memory is None:
                    # This case should ideally not happen if engines always return CompressedMemory
                    # Or handle as an error/empty input
                    # For now, let's assume it means stop processing or pass empty string
                    # This indicates a previous engine in the pipeline failed or returned None unexpectedly
                    # For robustness, we could log a warning and break, or try to continue with empty text.
                    # Let's assume for now that engines always return a valid CompressedMemory object.
                    # If current_compressed_memory is None here, it means an engine didn't return an object.
                    # This is a contract violation based on our new return types.
                    # However, to be safe, let's consider if we should raise an error or skip.
                    # For now, let's assume valid CompressedMemory objects are always passed.
                    pass  # Should not be reached if contracts are followed

                input_text_for_engine = current_compressed_memory.text
                prev_comp_result_for_engine = current_compressed_memory

            # Call the current engine's compress method
            current_compressed_memory = engine.compress(
                input_text_for_engine,
                budget,  # Pass the budget along
                previous_compression_result=prev_comp_result_for_engine,
                **kwargs,  # Pass other kwargs along
            )

            if current_compressed_memory and current_compressed_memory.trace:
                accumulated_traces.append(current_compressed_memory.trace)

        if current_compressed_memory is None:
            # This would happen if self.engines is empty or an engine returned None.
            # Create a default CompressedMemory if pipeline was empty or failed early.
            # For an empty pipeline, what should be returned? Perhaps the input text uncompressed?
            # Or an empty text with a trace indicating no operations?
            # For now, let's assume self.engines is not empty and all engines return valid CompressedMemory.
            # If it could be empty, we'd need to define behavior.
            # If the loop didn't run (empty self.engines), current_compressed_memory would be the initial previous_compression_result
            if (
                previous_compression_result
            ):  # Pipeline was empty, but initial result provided
                current_compressed_memory = previous_compression_result
            else:  # Pipeline was empty AND no initial result
                current_compressed_memory = CompressedMemory(
                    text=(
                        original_input_text
                        if isinstance(original_input_text, str)
                        else ""
                    ),
                    metadata={
                        "notes": "Pipeline was empty or no operations performed."
                    },
                )

        # Prepare strategy_params for the pipeline's trace
        pipeline_strategy_params = {"budget": budget, **kwargs}
        if isinstance(self.config, PipelineConfig):  # self.config is PipelineConfig
            pipeline_strategy_params["engines"] = [
                asdict(engine_conf) for engine_conf in self.config.engines
            ]
        elif isinstance(
            self.config, dict
        ):  # self.config could be a dict if BaseCompressionEngine.__init__ was used
            pipeline_strategy_params.update(self.config)

        # Create the pipeline's overall trace
        pipeline_trace = CompressionTrace(
            engine_name=self.id,
            strategy_params=pipeline_strategy_params,
            input_summary={
                "original_length": len(
                    original_input_text if isinstance(original_input_text, str) else ""
                )
            },
            steps=[
                asdict(trace) for trace in accumulated_traces
            ],  # Store sub-traces as dicts
            output_summary={"compressed_length": len(current_compressed_memory.text)},
            final_compressed_object_preview=current_compressed_memory.text[:50],
        )
        pipeline_trace.processing_ms = (time.monotonic() - start) * 1000

        # Create the final CompressedMemory object for the pipeline
        final_compressed_output = CompressedMemory(
            text=current_compressed_memory.text,
            engine_id=self.id,
            engine_config=pipeline_strategy_params,  # Or a cleaned version of self.config if preferred
            trace=pipeline_trace,
            metadata=current_compressed_memory.metadata,  # Preserve metadata from the last engine
        )

        return final_compressed_output


__all__ = ["EngineConfig", "PipelineConfig", "PipelineEngine"]

# Register this engine when imported so it is discoverable

register_compression_engine(
    PipelineEngine.id,
    PipelineEngine,
    display_name="Pipeline",
    source="built-in",
)
