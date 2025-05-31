"""Gist Memory Agent package."""

from .cli import app  # noqa: F401
from .models import (  # noqa: F401
    BeliefPrototype,
    RawMemory,
    ConversationalTurn,
)
from .json_npy_store import JsonNpyVectorStore, VectorStore  # noqa: F401
from .agent import Agent, QueryResult, PrototypeHit, MemoryHit  # noqa: F401
from .embedding_pipeline import embed_text  # noqa: F401
from .chunker import SentenceWindowChunker, FixedSizeChunker  # noqa: F401
from .config import DEFAULT_BRAIN_PATH  # noqa: F401
from .local_llm import LocalChatModel  # noqa: F401
from .canonical import render_five_w_template  # noqa: F401
from .memory_cues import MemoryCueRenderer  # noqa: F401
from .conflict_flagging import ConflictFlagger, ConflictLogger  # noqa: F401
from .experiment_runner import ExperimentConfig, run_experiment  # noqa: F401
from .history_experiment import (  # noqa: F401
    HistoryExperimentConfig,
    run_history_experiment,
)
from .conflict import SimpleConflictLogger, negation_conflict  # noqa: F401
from .talk_session import TalkSessionManager  # noqa: F401
from .utils import load_agent  # noqa: F401


# Exported symbols.  ``dict.fromkeys`` ensures each value only appears once
# while preserving the explicit ordering of the list below.
__all__ = list(
    dict.fromkeys(
        [
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
        ]
    )
)

# Semantic version of the package
__version__ = "0.1.0"
