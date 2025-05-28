"""Gist Memory Agent package."""

from .cli import cli
from .models import BeliefPrototype, RawMemory
from .json_npy_store import JsonNpyVectorStore, VectorStore
from .embedding_pipeline import embed_text
from .chunker import SentenceWindowChunker, FixedSizeChunker

__all__ = [
    "cli",
    "BeliefPrototype",
    "RawMemory",
    "JsonNpyVectorStore",
    "VectorStore",
    "embed_text",
    "SentenceWindowChunker",
    "FixedSizeChunker",
]

# Semantic version of the package
__version__ = "0.1.0"
