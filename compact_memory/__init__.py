"""Compact Memory package with lazy loading of submodules."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "BaseCompressionEngine",
    "CompressedMemory",
    "CompressionTrace",
    "PipelineEngineConfig",
    "PipelineEngine",
    "NoCompressionEngine",
    "register_compression_engine",
    "get_compression_engine",
    "available_engines",
    "get_engine_metadata",
    "all_engine_metadata",
]

_lazy_map = {
    "BaseCompressionEngine": "compact_memory.engines",
    "CompressedMemory": "compact_memory.engines",
    "CompressionTrace": "compact_memory.engines",
    "PipelineEngineConfig": "compact_memory.engines.pipeline_engine",
    "PipelineEngine": "compact_memory.engines.pipeline_engine",
    "NoCompressionEngine": "compact_memory.engines.no_compression_engine",
    "register_compression_engine": "compact_memory.engine_registry",
    "get_compression_engine": "compact_memory.engine_registry",
    "available_engines": "compact_memory.engine_registry",
    "get_engine_metadata": "compact_memory.engine_registry",
    "all_engine_metadata": "compact_memory.engine_registry",
}


def __getattr__(name: str) -> Any:  # pragma: no cover - simple passthrough
    if name in _lazy_map:
        module = importlib.import_module(_lazy_map[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - for completeness
    return sorted(list(globals().keys()) + list(_lazy_map.keys()))


__version__ = "1.0.0"
