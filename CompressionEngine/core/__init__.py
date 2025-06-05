"""Compression engine interfaces and implementations."""

from __future__ import annotations

from .engines_abc import CompressedMemory, CompressionEngine, CompressionTrace
from .config import EngineConfig
from .pipeline_strategy import PipelineCompressionEngine, PipelineEngineConfig
from .no_compression_engine import NoCompressionEngine
from .registry import (
    register_compression_engine,
    get_compression_engine,
    available_engines,
    get_engine_metadata,
    all_engine_metadata,
    get_engine_config_class,
    validate_engine_id,
)

register_compression_engine(NoCompressionEngine.id, NoCompressionEngine)
# TODO: When PipelineCompressionStrategy is renamed to PipelineCompressionEngine, update this registration.
# The class name PipelineCompressionEngine is already used, assuming it's the renamed class.
# If PipelineCompressionEngine has an associated config, it should be passed to config_cls.
register_compression_engine(
    PipelineCompressionEngine.id, PipelineCompressionEngine
)

__all__ = [
    "CompressedMemory",
    "CompressionEngine",
    "CompressionTrace",
    "NoCompressionEngine",
    "PipelineCompressionEngine",
    "PipelineEngineConfig",
    "register_compression_engine",
    "get_compression_engine",
    "available_engines",
    "get_engine_metadata",
    "all_engine_metadata",
    "get_engine_config_class",
    "validate_engine_id",
    "EngineConfig",
]
