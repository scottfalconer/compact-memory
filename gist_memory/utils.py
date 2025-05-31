from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from .json_npy_store import JsonNpyVectorStore
from .chunker import SentenceWindowChunker, _CHUNKER_REGISTRY
from .embedding_pipeline import embed_text, EmbeddingDimensionMismatchError


def get_disk_usage(path: Path) -> int:
    """Return total size of files under ``path`` in bytes."""
    size = 0
    for root, _, files in os.walk(path):
        for name in files:
            fp = Path(root) / name
            try:
                size += fp.stat().st_size
            except OSError:
                pass
    return size

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from .agent import Agent


def load_agent(path: Path) -> Agent:
    """Return an :class:`Agent` loaded from ``path``.

    If the stored embedding dimension does not match the current model,
    the store is re-initialized with the correct dimension.
    """
    from .agent import Agent

    try:
        store = JsonNpyVectorStore(path=str(path))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Agent directory '{path}' not found or is invalid"
        ) from exc
    except EmbeddingDimensionMismatchError:
        dim = int(embed_text(["dim"]).shape[1])
        store = JsonNpyVectorStore(path=str(path), embedding_dim=dim)

    chunker_id = store.meta.get("chunker", "sentence_window")
    chunker_cls = _CHUNKER_REGISTRY.get(chunker_id, SentenceWindowChunker)
    tau = float(store.meta.get("tau", 0.8))
    return Agent(store, chunker=chunker_cls(), similarity_threshold=tau)


__all__ = ["load_agent", "get_disk_usage"]
