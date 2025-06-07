from __future__ import annotations

"""Registry utilities for compression engines."""

from typing import Any, Dict, Optional, Type, List, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - hints only
    from .base import BaseCompressionEngine
else:  # pragma: no cover - provide dummy for type checkers

    class BaseCompressionEngine:  # type: ignore
        pass


# Import BaseCompressionEngine relatively for type hinting within this module if needed
# from .base import BaseCompressionEngine
# It seems BaseCompressionEngine is imported by modules that call register_compression_engine

_ENGINE_REGISTRY: Dict[str, Type["BaseCompressionEngine"]] = (
    {}
)  # Use string literal for forward ref if Base not imported
_ENGINE_INFO: Dict[str, Dict[str, Optional[str]]] = {}


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
    return _ENGINE_REGISTRY[id]


def available_engines() -> List[str]:
    return sorted(_ENGINE_REGISTRY)


def get_engine_metadata(id: str) -> Dict[str, Optional[str]] | None:
    info = _ENGINE_INFO.get(id)
    if info:
        info_with_id = info.copy()
        info_with_id["engine_id"] = id
        return info_with_id
    return None


def all_engine_metadata() -> Dict[str, Dict[str, Optional[str]]]:
    return dict(_ENGINE_INFO)


_BUILTIN_ENGINES_REGISTERED = False  # Guard to prevent multiple registrations.


def register_builtin_engines():
    """
    Registers all built-in compression engines.

    This function centralizes the registration of core engines like
    NoCompressionEngine and FirstLastEngine.
    It's designed to be called once, typically when the `compact_memory.engines`
    package is imported.
    """
    global _BUILTIN_ENGINES_REGISTERED
    if _BUILTIN_ENGINES_REGISTERED:
        return

    # Import engine classes here to minimize import side-effects at module load time
    # and to encapsulate these imports within the registration logic.
    from compact_memory.engines.no_compression_engine import NoCompressionEngine
    from compact_memory.engines.first_last_engine import FirstLastEngine

    register_compression_engine(
        NoCompressionEngine.id,
        NoCompressionEngine,
        display_name="No Compression",
        source="built-in",
    )
    register_compression_engine(
        FirstLastEngine.id,
        FirstLastEngine,
        display_name="First/Last Chunks",
        source="built-in",
    )
    _BUILTIN_ENGINES_REGISTERED = True


__all__ = [
    "register_compression_engine",
    "get_compression_engine",
    "available_engines",
    "get_engine_metadata",
    "all_engine_metadata",
    "register_builtin_engines",  # Expose the new function
]
