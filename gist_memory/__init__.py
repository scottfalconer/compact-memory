"""Gist Memory Agent package."""

from .cli import app
from .models import BeliefPrototype, RawMemory
from .json_npy_store import JsonNpyVectorStore, VectorStore
from .agent import Agent
from .embedding_pipeline import embed_text
from .chunker import SentenceWindowChunker, FixedSizeChunker

__all__ = [
    "app",
    "BeliefPrototype",
    "RawMemory",
    "JsonNpyVectorStore",
    "VectorStore",
    "Agent",
    "embed_text",
    "SentenceWindowChunker",
    "FixedSizeChunker",
]

# Semantic version of the package
__version__ = "0.1.0"
