"""Gist Memory Agent package."""

from .cli import app
from .models import BeliefPrototype, RawMemory
from .json_npy_store import JsonNpyVectorStore, VectorStore
from .agent import Agent, QueryResult, PrototypeHit, MemoryHit
from .embedding_pipeline import embed_text
from .chunker import SentenceWindowChunker, FixedSizeChunker
from .config import DEFAULT_BRAIN_PATH
from .local_llm import LocalChatModel
from .canonical import render_five_w_template
from .memory_cues import MemoryCueRenderer
from .conflict_flagging import ConflictFlagger, ConflictLogger
from .experiment_runner import ExperimentConfig, run_experiment
from .conflict import ConflictLogger, negation_conflict


__all__ = [
    "app",
    "BeliefPrototype",
    "RawMemory",
    "JsonNpyVectorStore",
    "VectorStore",
    "Agent",
    "QueryResult",
    "PrototypeHit",
    "MemoryHit",
    "embed_text",
    "SentenceWindowChunker",
    "FixedSizeChunker",
    "DEFAULT_BRAIN_PATH",
    "LocalChatModel",
    "render_five_w_template",
    "MemoryCueRenderer",
    "ConflictFlagger",
    "ConflictLogger",
    "ExperimentConfig",
    "run_experiment",
    "ConflictLogger",
    "negation_conflict",
]

# Semantic version of the package
__version__ = "0.1.0"
