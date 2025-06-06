"""Experimental compression engines and utilities."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "ActiveMemoryEngine",
    "FirstLastEngine",
    "ActiveMemoryManager",
    "ConversationTurn",
    "enable_all_experimental_engines",
]

_lazy_modules = {
    "ActiveMemoryEngine": "compact_memory.engines.active_memory_engine",
    "FirstLastEngine": "compact_memory.engines.first_last_engine",
    "ActiveMemoryManager": "compact_memory.active_memory_manager",
    "ConversationTurn": "compact_memory.active_memory_manager",
    'ReadAgentGistEngine': 'compact_memory.engines.ReadAgent.engine',
}


def __getattr__(name: str) -> Any:  # pragma: no cover - passthrough
    if name in _lazy_modules:
        module = importlib.import_module(_lazy_modules[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(list(globals().keys()) + list(_lazy_modules.keys()))


def enable_all_experimental_engines() -> None:
    """Register all experimental compression engines."""
    for module in _lazy_modules.values():
        try:
            importlib.import_module(module)
        except Exception:
            continue
