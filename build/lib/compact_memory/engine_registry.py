from __future__ import annotations

"""Top-level registry utilities for compression engines."""

from .engines.registry import (
    register_compression_engine,
    get_compression_engine,
    available_engines,
    get_engine_metadata,
    all_engine_metadata,
    _ENGINE_REGISTRY,
    _ENGINE_INFO,
)

__all__ = [
    "register_compression_engine",
    "get_compression_engine",
    "available_engines",
    "get_engine_metadata",
    "all_engine_metadata",
]
