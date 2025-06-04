"""Experimental compression strategies and utilities."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "ActiveMemoryStrategy",
    "ActiveMemoryManager",
    "ConversationTurn",
    "FirstLastStrategy",
    "ChainedStrategy",
    "RationaleEpisodeStrategy",
    "PrototypeSystemStrategy",
    "EvidenceWriter",
    "LLMSummarisingChunker",
]

_lazy_modules = {
    "ActiveMemoryStrategy": ".active_memory_strategy",
    "ActiveMemoryManager": ".active_memory_manager",
    "ConversationTurn": ".active_memory_manager",
    "FirstLastStrategy": ".first_last_strategy",
    "ChainedStrategy": ".chained",
    "RationaleEpisodeStrategy": ".rationale_episode",
    "PrototypeSystemStrategy": ".prototype_system",
    "EvidenceWriter": ".prototype_system",
    "LLMSummarisingChunker": ".llm_summarising_chunker",
}


def __getattr__(name: str) -> Any:  # pragma: no cover - passthrough
    if name in _lazy_modules:
        module = importlib.import_module(__name__ + _lazy_modules[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(list(globals().keys()) + list(_lazy_modules.keys()))
