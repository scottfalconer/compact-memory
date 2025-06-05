"""Compact Memory package with lazy loading of submodules."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "app",
    "BeliefPrototype",
    "RawMemory",
    "ConversationalTurn",
    "VectorStore",
    "InMemoryVectorStore",
    "MemoryContainer",
    "QueryResult",
    "PrototypeHit",
    "MemoryHit",
    "embed_text",
    "SentenceWindowChunker",
    "FixedSizeChunker",
    "AgenticChunker",
    "DEFAULT_MEMORY_PATH",
    "LocalChatModel",
    "LearnedSummarizerStrategy",
    "load_memory_container",
    "tokenize_text",
    "token_count",
    "PromptBudget",
    "CompressionStrategy",
    "NoCompression",
    "ValidationMetric",
    "StrategyConfig",
]

_lazy_map = {
    "app": "compact_memory.cli",
    "BeliefPrototype": "compact_memory.models",
    "RawMemory": "compact_memory.models",
    "ConversationalTurn": "compact_memory.models",
    "VectorStore": "compact_memory.vector_store",
    "InMemoryVectorStore": "compact_memory.vector_store",
    "MemoryContainer": "compact_memory.memory_container",
    "QueryResult": "compact_memory.memory_container",
    "PrototypeHit": "compact_memory.memory_container",
    "MemoryHit": "compact_memory.memory_container",
    "embed_text": "compact_memory.embedding_pipeline",
    "SentenceWindowChunker": "compact_memory.chunker",
    "FixedSizeChunker": "compact_memory.chunker",
    "AgenticChunker": "compact_memory.chunker",
    "DEFAULT_MEMORY_PATH": "compact_memory.config",
    "LocalChatModel": "compact_memory.local_llm",
    "load_memory_container": "compact_memory.utils",
    "tokenize_text": "compact_memory.token_utils",
    "token_count": "compact_memory.token_utils",
    "LearnedSummarizerStrategy": "CompressionStrategy.contrib.learned_summarizer_strategy",
    "PromptBudget": "compact_memory.prompt_budget",
    "CompressionStrategy": "CompressionStrategy.core",
    "NoCompression": "CompressionStrategy.core",
    "ValidationMetric": "compact_memory.validation.metrics_abc",
    "StrategyConfig": "CompressionStrategy.core",
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
