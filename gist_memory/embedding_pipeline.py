from __future__ import annotations

import functools
import hashlib
from typing import List, Sequence

import numpy as np

try:  # optional heavy deps
    import torch
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - during tests torch may be missing
    torch = None
    SentenceTransformer = None


class EmbeddingDimensionMismatchError(ValueError):
    """Raised when stored embedding dimension does not match model output."""


class MockEncoder:
    """Deterministic mock encoder used in tests."""

    dim = 32

    def encode(self, texts: List[str] | str, *args, **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        arr = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hashlib.sha256(t.encode("utf-8")).digest()
            arr[i] = np.frombuffer(h, dtype=np.uint8)[: self.dim] / 255.0
        if len(arr) == 1:
            return arr[0]
        return arr


_MODEL: SentenceTransformer | None = None
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_DEVICE = "cpu"
_BATCH_SIZE = 32


def _load_model(model_name: str, device: str) -> SentenceTransformer:
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers is required for embedding")
    global _MODEL, _MODEL_NAME, _DEVICE
    if _MODEL is None or model_name != _MODEL_NAME or device != _DEVICE:
        if torch is not None:
            torch.manual_seed(0)
        _MODEL = SentenceTransformer(model_name, device=device)
        _MODEL_NAME = model_name
        _DEVICE = device
    return _MODEL


@functools.lru_cache(maxsize=5000)
def _embed_cached(text: str, model_name: str, device: str, batch_size: int) -> np.ndarray:
    model = _load_model(model_name, device)
    vec = model.encode(text, batch_size=batch_size, convert_to_numpy=True)
    vec = vec.astype(np.float32)
    vec /= np.linalg.norm(vec) or 1.0
    return vec


def embed_text(
    text: str | Sequence[str],
    *,
    model_name: str = _MODEL_NAME,
    device: str = _DEVICE,
    batch_size: int = _BATCH_SIZE,
) -> np.ndarray:
    """Embed ``text`` or list of texts."""

    if isinstance(text, str):
        if text == "":
            model = _load_model(model_name, device)
            return np.zeros(model.get_sentence_embedding_dimension(), dtype=np.float32)
        return _embed_cached(text, model_name, device, batch_size)

    texts = list(text)
    if not texts:
        model = _load_model(model_name, device)
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    vecs = [_embed_cached(t, model_name, device, batch_size) for t in texts]
    return np.stack(vecs)


def register_embedding(name: str, encoder_callable) -> None:
    """Allow plugins to register alternative embedding functions."""
    globals()[name] = encoder_callable


__all__ = [
    "EmbeddingDimensionMismatchError",
    "MockEncoder",
    "embed_text",
    "register_embedding",
]

