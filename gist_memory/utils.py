from pathlib import Path

from .agent import Agent
from .json_npy_store import JsonNpyVectorStore
from .chunker import SentenceWindowChunker, _CHUNKER_REGISTRY
from .embedding_pipeline import embed_text, EmbeddingDimensionMismatchError


def load_agent(path: Path) -> Agent:
    """Return an :class:`Agent` loaded from ``path``.

    If the stored embedding dimension does not match the current model,
    the store is re-initialized with the correct dimension.
    """
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


__all__ = ["load_agent"]
