from __future__ import annotations

"""Registry utilities for compression engines."""

from typing import Any, Dict, Optional, Type, List, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from .base import BaseCompressionEngine

# Import BaseCompressionEngine relatively for type hinting within this module if needed
# from .base import BaseCompressionEngine
# It seems BaseCompressionEngine is imported by modules that call register_compression_engine

_ENGINE_REGISTRY: Dict[str, Type["BaseCompressionEngine"]] = (
    {}
)  # Use string literal for forward ref if Base not imported
_ENGINE_INFO: Dict[str, Dict[str, Optional[str]]] = {}


def _ensure_plugins_loaded() -> None:
    """Load plugins if they have not been loaded yet."""
    from compact_memory.plugin_loader import load_plugins

    load_plugins()


def register_compression_engine(
    id: str,
    cls: Type["BaseCompressionEngine"],  # Use string literal
    *,
    display_name: str | None = None,
    version: str | None = None,
    source: str = "built-in",
) -> None:
    """Register ``cls`` under ``id`` with optional metadata."""
    # Ensure cls is a subclass of a dynamically imported BaseCompressionEngine if not already checked
    # This is more for safety, actual type checking happens at usage points or via linters.
    # from .base import BaseCompressionEngine # Local import to avoid cycles if registry is imported by base
    # if not issubclass(cls, BaseCompressionEngine):
    #     raise TypeError(f"Class {cls.__name__} is not a subclass of BaseCompressionEngine")

    prev = _ENGINE_INFO.get(id)
    overrides = prev["source"] if prev else None
    _ENGINE_REGISTRY[id] = cls
    _ENGINE_INFO[id] = {
        "display_name": display_name or id,
        "version": version or "N/A",
        "source": source,
        "overrides": overrides,
    }


def get_compression_engine(
    id: str,
) -> Type["BaseCompressionEngine"]:  # Use string literal
    """Return the engine class registered under ``id``."""
    _ensure_plugins_loaded()
    return _ENGINE_REGISTRY[id]


def available_engines() -> List[str]:
    _ensure_plugins_loaded()
    return sorted(_ENGINE_REGISTRY)


def get_engine_metadata(id: str) -> Dict[str, Optional[str]] | None:
    _ensure_plugins_loaded()
    info = _ENGINE_INFO.get(id)
    if info:
        info_with_id = info.copy()
        info_with_id["engine_id"] = id
        return info_with_id
    return None


def all_engine_metadata() -> Dict[str, Dict[str, Optional[str]]]:
    _ensure_plugins_loaded()
    return dict(_ENGINE_INFO)


__all__ = [
    "register_compression_engine",
    "get_compression_engine",
    "available_engines",
    "get_engine_metadata",
    "all_engine_metadata",
]
