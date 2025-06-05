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
    "PrototypeEngine",
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
    "BaseCompressionEngine",
    "CompressedMemory",
    "CompressionTrace",
    "PipelineEngineConfig",
    "PipelineEngine",
    "NoCompressionEngine",
    "ValidationMetric",
    "StrategyConfig",
    "register_compression_engine",
    "get_compression_engine",
    "available_engines",
    "get_engine_metadata",
    "all_engine_metadata",
]

_lazy_map = {
    "app": "compact_memory.cli",
    "BeliefPrototype": "compact_memory.models",
    "RawMemory": "compact_memory.models",
    "ConversationalTurn": "compact_memory.models",
    "VectorStore": "compact_memory.vector_store",
    "InMemoryVectorStore": "compact_memory.vector_store",
    "PrototypeEngine": "compact_memory.prototype_engine",
    "QueryResult": "compact_memory.prototype_engine",
    "PrototypeHit": "compact_memory.prototype_engine",
    "MemoryHit": "compact_memory.prototype_engine",
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
    "BaseCompressionEngine": "compact_memory.engines",
    "CompressedMemory": "compact_memory.engines",
    "CompressionTrace": "compact_memory.engines",
    "PipelineEngineConfig": "compact_memory.engines.pipeline_engine",
    "PipelineEngine": "compact_memory.engines.pipeline_engine",
    "NoCompressionEngine": "compact_memory.engines.no_compression_engine",
    "ValidationMetric": "compact_memory.validation.metrics_abc",
    "StrategyConfig": "CompressionStrategy.core",
    "register_compression_engine": "compact_memory.engine_registry",
    "get_compression_engine": "compact_memory.engine_registry",
    "available_engines": "compact_memory.engine_registry",
    "get_engine_metadata": "compact_memory.engine_registry",
    "all_engine_metadata": "compact_memory.engine_registry",
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
