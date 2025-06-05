from __future__ import annotations

"""Registry utilities for compression engines."""

from typing import Any, Dict, Optional, Type, List

from . import BaseCompressionEngine

_ENGINE_REGISTRY: Dict[str, Type[BaseCompressionEngine]] = {}
_ENGINE_INFO: Dict[str, Dict[str, Optional[str]]] = {}


def register_compression_engine(
    id: str,
    cls: Type[BaseCompressionEngine],
    *,
    display_name: str | None = None,
    version: str | None = None,
    source: str = "built-in",
) -> None:
    """Register ``cls`` under ``id`` with optional metadata."""
    prev = _ENGINE_INFO.get(id)
    overrides = prev["source"] if prev else None
    _ENGINE_REGISTRY[id] = cls
    _ENGINE_INFO[id] = {
        "display_name": display_name or id,
        "version": version or "N/A",
        "source": source,
        "overrides": overrides,
    }


def get_compression_engine(id: str) -> Type[BaseCompressionEngine]:
    """Return the engine class registered under ``id``."""
    return _ENGINE_REGISTRY[id]


def available_engines() -> List[str]:
    return sorted(_ENGINE_REGISTRY)


def get_engine_metadata(id: str) -> Dict[str, Optional[str]] | None:
    info = _ENGINE_INFO.get(id)
    if info:
        # Make a copy and add the id to it
        info_with_id = info.copy()
        info_with_id["engine_id"] = id # Changed "id" to "engine_id"
        return info_with_id
    return None


def all_engine_metadata() -> Dict[str, Dict[str, Optional[str]]]:
    return dict(_ENGINE_INFO)


__all__ = [
    "register_compression_engine",
    "get_compression_engine",
    "available_engines",
    "get_engine_metadata",
    "all_engine_metadata",
]
