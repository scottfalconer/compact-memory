"""Gist Memory Agent package with lazy loading of submodules."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "app",
    "BeliefPrototype",
    "RawMemory",
    "ConversationalTurn",
    "JsonNpyVectorStore",
    "VectorStore",
    "Agent",
    "QueryResult",
    "PrototypeHit",
    "MemoryHit",
    "embed_text",
    "SentenceWindowChunker",
    "FixedSizeChunker",
    "AgenticChunker",
    "DEFAULT_BRAIN_PATH",
    "LocalChatModel",
    "render_five_w_template",
    "MemoryCueRenderer",
    "ConflictFlagger",
    "ConflictLogger",
    "SimpleConflictLogger",
    "ExperimentConfig",
    "run_experiment",
    "HistoryExperimentConfig",
    "run_history_experiment",
    "negation_conflict",
    "TalkSessionManager",
    "load_agent",
    "tokenize_text",
    "token_count",
    "PromptBudget",
    "ValidationMetric",
]

_lazy_map = {
    "app": "gist_memory.cli",
    "BeliefPrototype": "gist_memory.models",
    "RawMemory": "gist_memory.models",
    "ConversationalTurn": "gist_memory.models",
    "JsonNpyVectorStore": "gist_memory.json_npy_store",
    "VectorStore": "gist_memory.json_npy_store",
    "Agent": "gist_memory.agent",
    "QueryResult": "gist_memory.agent",
    "PrototypeHit": "gist_memory.agent",
    "MemoryHit": "gist_memory.agent",
    "embed_text": "gist_memory.embedding_pipeline",
    "SentenceWindowChunker": "gist_memory.chunker",
    "FixedSizeChunker": "gist_memory.chunker",
    "AgenticChunker": "gist_memory.chunker",
    "DEFAULT_BRAIN_PATH": "gist_memory.config",
    "LocalChatModel": "gist_memory.local_llm",
    "render_five_w_template": "gist_memory.prototype.canonical",
    "MemoryCueRenderer": "gist_memory.prototype.memory_cues",
    "ConflictFlagger": "gist_memory.prototype.conflict_flagging",
    "ConflictLogger": "gist_memory.prototype.conflict_flagging",
    "SimpleConflictLogger": "gist_memory.prototype.conflict",
    "ExperimentConfig": "gist_memory.experiment_runner",
    "run_experiment": "gist_memory.experiment_runner",
    "HistoryExperimentConfig": "gist_memory.history_experiment",
    "run_history_experiment": "gist_memory.history_experiment",
    "negation_conflict": "gist_memory.prototype.conflict",
    "TalkSessionManager": "gist_memory.talk_session",
    "load_agent": "gist_memory.utils",
    "tokenize_text": "gist_memory.token_utils",
    "token_count": "gist_memory.token_utils",
    "PromptBudget": "gist_memory.prompt_budget",
    "ValidationMetric": "gist_memory.validation.metrics_abc",
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


__version__ = "0.1.0"
