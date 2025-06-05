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
    # "MemoryContainer", # Removed
    # "QueryResult", # Removed
    # "PrototypeHit", # Removed
    # "MemoryHit", # Removed
    "embed_text",
    "SentenceWindowChunker",
    "FixedSizeChunker",
    "AgenticChunker",
    "DEFAULT_MEMORY_PATH", # This might change if config keys are updated
    "LocalChatModel",
    "SummarizationEngine", # Changed from LearnedSummarizerStrategy
    "load_memory_container", # This function might be obsolete if MemoryContainer is removed
    "tokenize_text",
    "token_count",
    "PromptBudget",
    "CompressionEngine", # Changed from CompressionStrategy
    "NoCompressionEngine", # Changed from NoCompression
    "ValidationMetric",
    "EngineConfig", # Changed from StrategyConfig
]

_lazy_map = {
    "app": "compact_memory.cli",
    "BeliefPrototype": "compact_memory.models",
    "RawMemory": "compact_memory.models",
    "ConversationalTurn": "compact_memory.models",
    "VectorStore": "compact_memory.vector_store",
    "InMemoryVectorStore": "compact_memory.vector_store",
    # "MemoryContainer": "compact_memory.memory_container", # Removed
    # "QueryResult": "compact_memory.memory_container", # Removed
    # "PrototypeHit": "compact_memory.memory_container", # Removed
    # "MemoryHit": "compact_memory.memory_container", # Removed
    "embed_text": "compact_memory.embedding_pipeline",
    "SentenceWindowChunker": "compact_memory.chunker",
    "FixedSizeChunker": "compact_memory.chunker",
    "AgenticChunker": "compact_memory.chunker",
    "DEFAULT_MEMORY_PATH": "compact_memory.config", # This might change if config keys are updated
    "LocalChatModel": "compact_memory.local_llm",
    "load_memory_container": "compact_memory.utils", # This function might be obsolete
    "tokenize_text": "compact_memory.token_utils",
    "token_count": "compact_memory.token_utils",
    "SummarizationEngine": "CompressionEngine.contrib.summarization_engine", # Updated path and name
    "PromptBudget": "compact_memory.prompt_budget",
    "CompressionEngine": "CompressionEngine.core", # Updated path and name
    "NoCompressionEngine": "CompressionEngine.core.no_compression_engine", # Updated path and name
    "ValidationMetric": "compact_memory.validation.metrics_abc",
    "EngineConfig": "CompressionEngine.core.config", # Updated path and name
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
