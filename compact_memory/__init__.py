"""Compact Memory Agent package with lazy loading of submodules."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "app",
    "BeliefPrototype",
    "RawMemory",
    "ConversationalTurn",
    "MemoryStore",
    "BaseVectorStore",
    "InMemoryVectorStore",
    "Agent",
    "QueryResult",
    "PrototypeHit",
    "MemoryHit",
    "embed_text",
    "DEFAULT_MEMORY_PATH",
    "LocalChatModel",
    "render_five_w_template",
    "MemoryCueRenderer",
    "ConflictFlagger",
    "ConflictLogger",
    "SimpleConflictLogger",
    "LearnedSummarizerStrategy",
    "negation_conflict",
    "load_agent",
    "tokenize_text",
    "token_count",
    "PromptBudget",
    "CompressionStrategy",
    "NoCompression",
    "ImportanceCompression",
    "PrototypeSystemStrategy",
    "StrategyConfig",
]

_lazy_map = {
    "app": "compact_memory.cli",
    "BeliefPrototype": "compact_memory.models",
    "RawMemory": "compact_memory.models",
    "ConversationalTurn": "compact_memory.models",
    "MemoryStore": "compact_memory.memory_store",
    "BaseVectorStore": "compact_memory.vector_store",
    "InMemoryVectorStore": "compact_memory.vector_store",
    "Agent": "compact_memory.agent",
    "QueryResult": "compact_memory.agent",
    "PrototypeHit": "compact_memory.agent",
    "MemoryHit": "compact_memory.agent",
    "embed_text": "compact_memory.embedding_pipeline",
    "DEFAULT_MEMORY_PATH": "compact_memory.config",
    "LocalChatModel": "compact_memory.local_llm",
    "render_five_w_template": "compact_memory.prototype.canonical",
    "MemoryCueRenderer": "compact_memory.prototype.memory_cues",
    "ConflictFlagger": "compact_memory.prototype.conflict_flagging",
    "ConflictLogger": "compact_memory.prototype.conflict_flagging",
    "SimpleConflictLogger": "compact_memory.prototype.conflict",
    "negation_conflict": "compact_memory.prototype.conflict",
    "load_agent": "compact_memory.utils",
    "tokenize_text": "compact_memory.token_utils",
    "token_count": "compact_memory.token_utils",
    "LearnedSummarizerStrategy": "compact_memory.learned_summarizer_strategy",
    "PromptBudget": "compact_memory.prompt_budget",
    "CompressionStrategy": "compact_memory.compression",
    "NoCompression": "compact_memory.compression",
    "ImportanceCompression": "compact_memory.compression",
    "PrototypeSystemStrategy": "compact_memory.prototype_system_strategy",
    "StrategyConfig": "compact_memory.compression",
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
