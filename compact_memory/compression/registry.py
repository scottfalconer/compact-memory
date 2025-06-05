from __future__ import annotations

"""Registration utilities for compression strategies."""

from typing import Dict, Optional as Opt, Type, List, Any

from .strategies_abc import CompressionStrategy

_COMPRESSION_REGISTRY: Dict[str, Type[CompressionStrategy]] = {}
_COMPRESSION_INFO: Dict[str, Dict[str, Opt[str]]] = {}


def register_compression_strategy(
    id: str,
    cls: Type[CompressionStrategy],
    *,
    display_name: str | None = None,
    version: str | None = None,
    source: str = "built-in",
) -> None:
    """Register ``cls`` under ``id`` with optional metadata."""
    prev = _COMPRESSION_INFO.get(id)
    overrides = prev["source"] if prev else None
    _COMPRESSION_REGISTRY[id] = cls
    _COMPRESSION_INFO[id] = {
        "display_name": display_name or id,
        "version": version or "N/A",
        "source": source,
        "overrides": overrides,
    }


def get_compression_strategy(id: str) -> Type[CompressionStrategy]:
    """Return the CompressionStrategy class registered under ``id``."""
    return _COMPRESSION_REGISTRY[id]


def available_strategies() -> List[str]:
    return sorted(_COMPRESSION_REGISTRY)


def get_strategy_metadata(id: str) -> Dict[str, Opt[str]] | None:
    return _COMPRESSION_INFO.get(id)


def all_strategy_metadata() -> Dict[str, Dict[str, Opt[str]]]:
    return dict(_COMPRESSION_INFO)


__all__ = [
    "register_compression_strategy",
    "get_compression_strategy",
    "available_strategies",
    "get_strategy_metadata",
    "all_strategy_metadata",
]
