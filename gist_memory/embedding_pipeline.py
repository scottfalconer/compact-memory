from __future__ import annotations

import functools
import hashlib
from typing import List, Sequence

import tiktoken
from .importance_filter import dynamic_importance_filter

# ---------------------------------------------------------------------------
# Disable tqdm's multiprocessing lock.
# On some platforms (notably macOS with the "spawn" start method) creating
# the default multiprocessing lock can fail with "ValueError: bad value(s) in
# fds_to_keep".  Avoid this by monkeypatching tqdm to skip the mp lock.
try:  # pragma: no cover - only needed on certain platforms
    import tqdm.std

    def _no_mp_lock(cls):
        cls.mp_lock = None

    tqdm.std.TqdmDefaultWriteLock.create_mp_lock = classmethod(_no_mp_lock)
except Exception:  # pragma: no cover - tqdm not installed or different API
    pass

import numpy as np

# Lazy heavy imports will occur in ``_load_model``
torch = None  # type: ignore
SentenceTransformer = None  # type: ignore


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
    global _MODEL, _MODEL_NAME, _DEVICE, SentenceTransformer, torch
    if SentenceTransformer is None:
        try:
            from sentence_transformers import SentenceTransformer as _ST
            SentenceTransformer = _ST
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "sentence-transformers is required for embedding"
            ) from exc
    if torch is None:
        try:
            import torch as _torch
            torch = _torch  # type: ignore
        except Exception:  # pragma: no cover - torch not installed
            torch = None  # type: ignore

    if _MODEL is None or model_name != _MODEL_NAME or device != _DEVICE:
        if torch is not None:
            torch.manual_seed(0)
        try:
            _MODEL = SentenceTransformer(model_name, device=device)
        except Exception as exc:  # pragma: no cover - depends on local files
            raise RuntimeError(
                f"Error: Embedding Model '{model_name}' not found. "
                f"Please run: gist-memory download-model --model-name {model_name} to install it."
            ) from exc
        _MODEL_NAME = model_name
        _DEVICE = device
    return _MODEL


@functools.lru_cache(maxsize=5000)
def _embed_cached(
    text: str, model_name: str, device: str, batch_size: int
) -> np.ndarray:
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

    try:
        tokenizer = tiktoken.get_encoding("gpt2")
    except Exception:  # pragma: no cover - fallback if tokenizer missing
        tokenizer = None

    def _too_long(t: str) -> bool:
        if tokenizer is not None:
            try:
                return len(tokenizer.encode(t)) > 1000
            except Exception:  # pragma: no cover - encoding failure
                return len(t.split()) > 1000
        return len(t.split()) > 1000

    if isinstance(text, str):
        if text == "":
            model = _load_model(model_name, device)
            return np.zeros(model.get_sentence_embedding_dimension(), dtype=np.float32)
        if _too_long(text):
            text = dynamic_importance_filter(text)
        return _embed_cached(text, model_name, device, batch_size)

    texts = list(text)
    if not texts:
        model = _load_model(model_name, device)
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    filtered = []
    for t in texts:
        if _too_long(t):
            t = dynamic_importance_filter(t)
        filtered.append(t)
    vecs = [_embed_cached(t, model_name, device, batch_size) for t in filtered]
    return np.stack(vecs)


def get_embedding_dim(model_name: str = _MODEL_NAME, device: str = _DEVICE) -> int:
    """Return the embedding dimension for ``model_name`` on ``device``."""

    model = _load_model(model_name, device)
    get_dim = getattr(model, "get_sentence_embedding_dimension", None)
    if callable(get_dim):
        return int(get_dim())
    dim = getattr(model, "dim", None)
    if isinstance(dim, int):
        return dim
    raise AttributeError("embedding dimension not found")


def register_embedding(name: str, encoder_callable) -> None:
    """Allow plugins to register alternative embedding functions."""
    globals()[name] = encoder_callable


__all__ = [
    "EmbeddingDimensionMismatchError",
    "MockEncoder",
    "embed_text",
    "get_embedding_dim",
    "register_embedding",
]
