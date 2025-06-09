from __future__ import annotations

"""Pipeline compression engine for chaining multiple engines."""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union
import logging
import time

from .base import BaseCompressionEngine, CompressedMemory, CompressionTrace
from compact_memory.engine_config import EngineConfig as BaseEngineConfig
from .registry import get_compression_engine, register_compression_engine

# Re-export EngineConfig from this module for backward compatibility
EngineConfig = BaseEngineConfig


@dataclass
class PipelineStepConfig:  # Renamed local EngineConfig to avoid collision
    """Configuration for creating a :class:`BaseCompressionEngine` for a pipeline step."""

    engine_name: str
    engine_params: Dict[str, Any] = field(default_factory=dict)  # Changed dict to Dict

    def create(
        self,
    ) -> (
        BaseCompressionEngine
    ):  # Ensure BaseCompressionEngine is the correct return type
        cls = get_compression_engine(self.engine_name)
        # Correctly pass 'config' if it's part of engine_params and is BaseEngineConfig
        engine_config_params = self.engine_params.get("config")
        other_params = {k: v for k, v in self.engine_params.items() if k != "config"}
        return cls(config=engine_config_params, **other_params)


@dataclass
class PipelineConfig:  # This is the config for the PipelineEngine's structure
    """Configuration for :class:`PipelineEngine`."""

    engines: List[PipelineStepConfig] = field(
        default_factory=list
    )  # Uses renamed PipelineStepConfig

    def create(self) -> "PipelineEngine":  # Forward reference to PipelineEngine
        return PipelineEngine(self)


class PipelineEngine(BaseCompressionEngine):
    """Execute a sequence of other compression engines."""

    id = "pipeline"
    pipeline_structure_config: PipelineConfig  # Class attribute hint
    engines: List[BaseCompressionEngine]  # Class attribute hint

    def __init__(
        self,
        pipeline_definition: (
            Union[PipelineConfig, List[BaseCompressionEngine]] | None
        ) = None,
        *,
        config_or_engines: Union[
            PipelineConfig, List[BaseCompressionEngine], None
        ] = None,
        config: Optional[BaseEngineConfig | Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, **kwargs)

        if pipeline_definition is None:
            pipeline_definition = (
                config_or_engines if config_or_engines is not None else []
            )

        self.engines = []
        if isinstance(pipeline_definition, PipelineConfig):
            self.pipeline_structure_config = PipelineConfig(engines=[])
            for step in pipeline_definition.engines:
                if isinstance(step, PipelineStepConfig):
                    self.pipeline_structure_config.engines.append(step)
                    self.engines.append(step.create())
                elif isinstance(step, BaseEngineConfig):
                    self.pipeline_structure_config.engines.append(
                        PipelineStepConfig(step.engine_name, {"config": step})
                    )
                    self.engines.append(
                        get_compression_engine(step.engine_name)(config=step)
                    )
                elif isinstance(step, BaseCompressionEngine):
                    self.pipeline_structure_config.engines.append(
                        PipelineStepConfig(step.id, {"config": step.config})
                    )
                    self.engines.append(step)
                else:
                    raise TypeError("Unsupported engine specification in pipeline")
        else:
            self.pipeline_structure_config = PipelineConfig(engines=[])
            for item in pipeline_definition:
                if isinstance(item, BaseCompressionEngine):
                    self.engines.append(item)
                    params = {}
                    if isinstance(item.config, BaseEngineConfig):
                        params = item.config.model_dump(exclude_defaults=True)
                    self.pipeline_structure_config.engines.append(
                        PipelineStepConfig(item.id, params)
                    )
                elif isinstance(item, BaseEngineConfig):
                    self.pipeline_structure_config.engines.append(
                        PipelineStepConfig(item.engine_name, {"config": item})
                    )
                    self.engines.append(
                        get_compression_engine(item.engine_name)(config=item)
                    )
                elif isinstance(item, PipelineStepConfig):
                    self.pipeline_structure_config.engines.append(item)
                    self.engines.append(item.create())
                else:
                    raise TypeError("Unsupported engine specification in pipeline list")

    def compress(
        self,
        text_or_chunks: Union[
            str, List[str]
        ],  # This is the original text for the first engine if no previous_compression_result
        budget: int,
        previous_compression_result: Optional[CompressedMemory] = None,
        **kwargs: Any,
    ) -> CompressedMemory:
        """
        Executes a sequence of compression engines.

        Each engine in the pipeline processes the output of the previous one.
        The `budget` and other `kwargs` are passed to each engine in the sequence.

        Trace generation for the pipeline itself is controlled by this PipelineEngine's
        `self.config.enable_trace`. If False, the overall pipeline trace will be None.
        Individual sub-engines will generate their own traces based on their respective
        configurations, and these are included in the pipeline's trace if it's enabled.
        """
        # Need to import logging at the top of the file if not already there.
        # For this diff, assuming it will be added or is present.
        # import logging
        # Ensure self.config (the EngineConfig from BaseCompressionEngine) is used for enable_trace.
        # The __init__ was refactored to ensure super().__init__(config=config, **kwargs) sets this up.

        logging.debug(
            f"PipelineEngine '{self.id}': Starting compression pipeline, budget: {budget}, enable_trace for pipeline: {self.config.enable_trace}"
        )
        start_time = time.monotonic()

        current_compressed_memory: Optional[CompressedMemory] = (
            previous_compression_result
        )
        accumulated_traces: List[CompressionTrace] = []

        original_input_text = text_or_chunks
        if isinstance(
            original_input_text, List
        ):  # Ensure original_input_text is a string for len()
            original_input_text_str = " ".join(original_input_text)
        else:
            original_input_text_str = original_input_text

        for i, engine_instance in enumerate(self.engines):
            input_text_for_engine: str
            prev_comp_result_for_engine: Optional[CompressedMemory] = (
                current_compressed_memory
            )

            if i == 0 and current_compressed_memory is None:
                input_text_for_engine = original_input_text_str
            elif current_compressed_memory is not None:
                input_text_for_engine = current_compressed_memory.text
            else:
                logging.warning(
                    f"PipelineEngine '{self.id}': No current compressed memory to process for engine {engine_instance.id}, stopping pipeline."
                )
                break

            logging.debug(
                f"PipelineEngine '{self.id}': Running sub-engine '{engine_instance.id}'. Input text length: {len(input_text_for_engine)}"
            )
            current_compressed_memory = engine_instance.compress(
                input_text_for_engine,
                budget,
                previous_compression_result=prev_comp_result_for_engine,
                **kwargs,
            )

            if current_compressed_memory and current_compressed_memory.trace:
                # Sub-engine's trace is only added if the sub-engine itself had tracing enabled.
                accumulated_traces.append(current_compressed_memory.trace)
            elif (
                not current_compressed_memory
            ):  # Engine returned None, critical failure
                # Create a minimal CompressedMemory to signify failure at this stage
                current_compressed_memory = CompressedMemory(
                    text=(
                        prev_comp_result_for_engine.text
                        if prev_comp_result_for_engine
                        else ""
                    ),
                    engine_id=self.id,
                    engine_config=self.config.model_dump(
                        mode="json"
                    ),  # This PipelineEngine's config
                    trace=None,  # No pipeline trace if a sub-engine fails critically
                    metadata={
                        "error": f"Engine {engine_instance.id} failed to return CompressedMemory."
                    },
                )
                # If we want to stop the pipeline on such failure:
                # warnings.warn(f"Engine {engine_instance.id} in pipeline failed. Stopping pipeline.", RuntimeWarning)
                # return current_compressed_memory
                # For now, we continue, and the error will be in the metadata of the last successful state.
                # Or, more simply, the pipeline's trace will just be shorter.
                # The code below handles current_compressed_memory being None if all engines fail or list is empty.

        # Handle cases where the pipeline might be empty or all sub-engines failed to produce output
        if current_compressed_memory is None:
            final_text_output = (
                original_input_text_str
                if previous_compression_result is None
                else previous_compression_result.text
            )
            current_compressed_memory = CompressedMemory(
                text=final_text_output,
                metadata={"notes": "Pipeline was empty or no operations performed."},
            )

        # Now, check if the PipelineEngine itself should produce a trace
        if (
            not self.config.enable_trace
        ):  # self.config is EngineConfig from BaseCompressionEngine
            logging.debug(
                f"PipelineEngine '{self.id}': Tracing disabled for the pipeline itself. Skipping overall trace generation."
            )
            config_dump = self.config.model_dump(mode="json")
            config_dump.update({"budget": budget})
            if hasattr(self, "pipeline_structure_config"):
                config_dump["engines"] = [
                    asdict(ecfg) for ecfg in self.pipeline_structure_config.engines
                ]
            return CompressedMemory(
                text=current_compressed_memory.text,
                trace=None,
                engine_id=self.id,
                engine_config=config_dump,
                metadata=current_compressed_memory.metadata,
            )

        # Proceed with creating the pipeline's trace
        logging.debug(
            f"PipelineEngine '{self.id}': Generating overall pipeline trace. Number of sub-traces collected: {len(accumulated_traces)}"
        )
        pipeline_strategy_params = {"budget": budget, **kwargs}
        # Use self.pipeline_structure_config for tracing the structure
        if (
            hasattr(self, "pipeline_structure_config")
            and self.pipeline_structure_config.engines
        ):
            pipeline_strategy_params["engines"] = [
                asdict(engine_conf)
                for engine_conf in self.pipeline_structure_config.engines
            ]
        else:  # Fallback if instantiated with list of engines and structure wasn't fully parsable
            pipeline_strategy_params["engines"] = [eng.id for eng in self.engines]

        pipeline_trace = CompressionTrace(
            engine_name=self.id,
            strategy_params=pipeline_strategy_params,
            input_summary={"original_length": len(original_input_text_str)},
            steps=[
                asdict(trace) for trace in accumulated_traces
            ],  # accumulated_traces contains traces from sub-engines that had tracing enabled
            output_summary={"compressed_length": len(current_compressed_memory.text)},
            final_compressed_object_preview=current_compressed_memory.text[:50],
        )
        pipeline_trace.processing_ms = (time.monotonic() - start_time) * 1000

        config_dump = self.config.model_dump(mode="json")
        config_dump.update({"budget": budget})
        config_dump["engines"] = [
            asdict(e) for e in self.pipeline_structure_config.engines
        ]
        return CompressedMemory(
            text=current_compressed_memory.text,
            engine_id=self.id,
            engine_config=config_dump,
            trace=pipeline_trace,
            metadata=current_compressed_memory.metadata,
        )


__all__ = ["EngineConfig", "PipelineConfig", "PipelineEngine"]

# Register this engine when imported so it is discoverable

register_compression_engine(
    PipelineEngine.id,
    PipelineEngine,
    display_name="Pipeline",
    source="built-in",
)
