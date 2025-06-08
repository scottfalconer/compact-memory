"""Compact Memory package with lazy loading of submodules."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "BaseCompressionEngine",
    "CompressedMemory",
    "CompressionTrace",
    "PipelineConfig",
    "PipelineEngine",
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
    "PipelineConfig": "compact_memory.engines.pipeline_engine",
    "PipelineEngine": "compact_memory.engines.pipeline_engine",
    "register_compression_engine": "compact_memory.engines.registry",
    "get_compression_engine": "compact_memory.engines.registry",
    "available_engines": "compact_memory.engines.registry",
    "get_engine_metadata": "compact_memory.engines.registry",
    "all_engine_metadata": "compact_memory.engines.registry",
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


__version__ = "0.1.0"
