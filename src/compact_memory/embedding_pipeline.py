from __future__ import annotations

import functools
import hashlib
import contextlib
from typing import (
    List,
    Sequence,
    Callable,
    Optional,
    Any,
    Iterator,
)  # Changed ContextManager to Iterator

import warnings

from . import token_utils

import tiktoken

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
import os
import openai

# Lazy heavy imports will occur in ``_load_model``
torch = None  # type: ignore
SentenceTransformer = None  # type: ignore


# EmbeddingDimensionMismatchError moved to compact_memory.exceptions


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


_MODEL: Optional[Any] = None  # Changed SentenceTransformer | None to Optional[Any]
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_DEVICE = "cpu"
_BATCH_SIZE = 32

# Known dimensions for OpenAI embedding models
_OPENAI_EMBED_DIMS = {
    "text-embedding-ada-002": 1536,
}


def _load_model(model_name: str, device: str) -> Any:  # Changed return type
    global _MODEL, _MODEL_NAME, _DEVICE, SentenceTransformer, torch
    if SentenceTransformer is None:
        try:
            from sentence_transformers import SentenceTransformer as _ST

            SentenceTransformer = _ST
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "sentence-transformers is required for embedding. "
                "Install with 'pip install \"compact-memory[embedding]\"'."
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
            _MODEL = SentenceTransformer(
                model_name,
                device=device,
                local_files_only=True,
            )
        except Exception as exc:  # pragma: no cover - depends on local files
            raise RuntimeError(
                f"Error: Embedding Model '{model_name}' not found. "
                f"Please run: compact-memory download-model --model-name {model_name} to install it."
            ) from exc
        _MODEL_NAME = model_name
        _DEVICE = device
    return _MODEL


def load_model(model_name: str = _MODEL_NAME, device: str = _DEVICE) -> None:
    """Explicitly load the embedding model."""
    _load_model(model_name, device)


def unload_model() -> None:
    """Unload the cached embedding model to free memory."""
    global _MODEL
    _MODEL = None


@contextlib.contextmanager
def loaded_embedding_model(
    model_name: str = _MODEL_NAME, device: str = _DEVICE
) -> Iterator[Any]:  # Changed return type to Iterator[Any]
    """Context manager that loads and unloads the embedding model."""
    model: Optional[Any] = _load_model(model_name, device)
    try:
        yield model
    finally:
        unload_model()


@functools.lru_cache(maxsize=5000)
def _embed_cached(
    text: str, model_name: str, device: str, batch_size: int
) -> np.ndarray:
    if model_name.startswith("openai/"):
        base = model_name.split("/", 1)[1]
        client_kwargs: dict[str, str] = {}
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        client = openai.OpenAI(**client_kwargs)
        resp = client.embeddings.create(input=[text], model=base)
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        vec /= np.linalg.norm(vec) or 1.0
        return vec
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
    preprocess_fn: Callable[[str], str] | None = None,
) -> np.ndarray:
    """Embed ``text`` or list of texts."""

    try:
        tiktoken.get_encoding("gpt2")
    except Exception:  # pragma: no cover - fallback if tokenizer missing
        pass

    if isinstance(text, str):
        if text == "":
            if model_name.startswith("openai/"):
                base = model_name.split("/", 1)[1]
                dim = _OPENAI_EMBED_DIMS.get(base, 1536)
                return np.zeros(dim, dtype=np.float32)
            model: Optional[Any] = _load_model(model_name, device)
            if model and hasattr(model, "get_sentence_embedding_dimension"):
                return np.zeros(
                    model.get_sentence_embedding_dimension(), dtype=np.float32
                )
            # Fallback if model or method is not available, though get_embedding_dim might be better
            return np.zeros(get_embedding_dim(model_name, device), dtype=np.float32)
        if preprocess_fn is not None:
            text = preprocess_fn(text)

        tokenizer = None
        max_len = None
        try:
            if model_name.startswith("openai/"):
                base = model_name.split("/", 1)[1]
                tokenizer = tiktoken.encoding_for_model(base)
                max_len = getattr(tokenizer, "model_max_length", 8191)
            else:
                try:
                    from transformers import AutoTokenizer
                except Exception:  # pragma: no cover - transformers optional
                    from .local_llm import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(model_name)
                max_len = getattr(tokenizer, "model_max_length", None)
        except Exception:
            tokenizer = None

        if tokenizer is not None and isinstance(max_len, int) and max_len > 0:
            try:
                tokens = token_utils.token_count(tokenizer, text)
            except Exception:
                tokens = len(text.split())
            if tokens > max_len:
                warnings.warn(
                    f"Input exceeds model_max_length for {model_name}; embedding in segments",
                    RuntimeWarning,
                )
                parts = token_utils.split_by_tokens(tokenizer, text, max_len)
                vecs = [_embed_cached(p, model_name, device, batch_size) for p in parts]
                return np.stack(vecs).mean(axis=0)

        return _embed_cached(text, model_name, device, batch_size)

    texts = list(text)
    if not texts:
        if model_name.startswith("openai/"):
            base = model_name.split("/", 1)[1]
            dim = _OPENAI_EMBED_DIMS.get(base, 1536)
            return np.zeros((0, dim), dtype=np.float32)
        model: Optional[Any] = _load_model(model_name, device)
        if model and hasattr(model, "get_sentence_embedding_dimension"):
            return np.zeros(
                (0, model.get_sentence_embedding_dimension()), dtype=np.float32
            )
        return np.zeros((0, get_embedding_dim(model_name, device)), dtype=np.float32)
    if preprocess_fn is not None:
        texts = [preprocess_fn(t) for t in texts]

    tokenizer = None
    max_len = None
    try:
        if model_name.startswith("openai/"):
            base = model_name.split("/", 1)[1]
            tokenizer = tiktoken.encoding_for_model(base)
            max_len = getattr(tokenizer, "model_max_length", 8191)
        else:
            try:
                from transformers import AutoTokenizer
            except Exception:  # pragma: no cover - transformers optional
                from .local_llm import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            max_len = getattr(tokenizer, "model_max_length", None)
    except Exception:
        tokenizer = None

    if tokenizer is not None and isinstance(max_len, int) and max_len > 0:
        processed: List[np.ndarray] = []
        for t in texts:
            try:
                tokens = token_utils.token_count(tokenizer, t)
            except Exception:
                tokens = len(t.split())
            if tokens > max_len:
                warnings.warn(
                    f"Input exceeds model_max_length for {model_name}; embedding in segments",
                    RuntimeWarning,
                )
                parts = token_utils.split_by_tokens(tokenizer, t, max_len)
                vecs = [_embed_cached(p, model_name, device, batch_size) for p in parts]
                processed.append(np.stack(vecs).mean(axis=0))
            else:
                processed.append(_embed_cached(t, model_name, device, batch_size))
        return np.stack(processed)

    vecs = [_embed_cached(t, model_name, device, batch_size) for t in texts]
    return np.stack(vecs)


def get_embedding_dim(model_name: str = _MODEL_NAME, device: str = _DEVICE) -> int:
    """Return the embedding dimension for ``model_name`` on ``device``."""

    if model_name.startswith("openai/"):
        base = model_name.split("/", 1)[1]
        return _OPENAI_EMBED_DIMS.get(base, 1536)

    loaded_model_obj: Optional[Any] = _load_model(
        model_name, device
    )  # Renamed 'model' to 'loaded_model_obj'
    if loaded_model_obj:  # Ensure loaded_model_obj is not None
        get_dim_func = getattr(
            loaded_model_obj, "get_sentence_embedding_dimension", None
        )
        if callable(get_dim_func):
            return int(get_dim_func())
        dim_attr = getattr(loaded_model_obj, "dim", None)
        if isinstance(dim_attr, int):
            return dim_attr
    # Fallback or raise error if loaded_model_obj is None or attributes are missing
    raise AttributeError(
        f"Could not determine embedding dimension for model {model_name}. Model object: {loaded_model_obj}"
    )


def register_embedding(
    name: str, encoder_callable: Callable
) -> None:  # Added Callable type hint
    """Allow plugins to register alternative embedding functions."""
    globals()[name] = encoder_callable


__all__ = [
    # "EmbeddingDimensionMismatchError" # Removed from here
    "MockEncoder",
    "load_model",
    "unload_model",
    "loaded_embedding_model",
    "embed_text",
    "get_embedding_dim",
    "register_embedding",
]
