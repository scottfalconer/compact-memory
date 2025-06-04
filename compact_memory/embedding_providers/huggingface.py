from __future__ import annotations

import functools
import contextlib
from typing import Sequence, Union # Union was used in original embed_text

import numpy as np

# Tiktoken and dynamic_importance_filter are used by the original embed_text logic
# If they are strictly part of the HF embedding process, they belong here.
# If they are general text pre-processing, they might belong elsewhere or be passed in.
# For now, moving them here as they were coupled with the HF embedding logic.
try:
    import tiktoken
except ImportError: # pragma: no cover
    tiktoken = None # type: ignore

from ..importance_filter import dynamic_importance_filter # Relative import from parent package

# Lazy heavy imports will occur in ``_load_model``
torch = None  # type: ignore
SentenceTransformer = None  # type: ignore

# Default model parameters, can be overridden by arguments to functions
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DEVICE = "cpu"
DEFAULT_BATCH_SIZE = 32

# Module-level globals to cache the loaded model
_MODEL: SentenceTransformer | None = None
_MODEL_NAME_LOADED: str | None = None # Store name of currently loaded model
_DEVICE_LOADED: str | None = None   # Store device of currently loaded model


def _load_model(model_name: str, device: str) -> SentenceTransformer:
    """Loads the SentenceTransformer model if not already loaded or if params changed."""
    global _MODEL, _MODEL_NAME_LOADED, _DEVICE_LOADED, SentenceTransformer, torch

    if SentenceTransformer is None:
        try:
            from sentence_transformers import SentenceTransformer as _ST
            SentenceTransformer = _ST
        except ImportError as exc: # pragma: no cover
            raise ImportError(
                "sentence-transformers is required for Hugging Face embeddings. "
                "Please install it with `pip install sentence-transformers`."
            ) from exc

    if torch is None: # Optional, but SentenceTransformer uses it
        try:
            import torch as _torch
            torch = _torch
        except ImportError: # pragma: no cover
            pass # PyTorch not installed, SentenceTransformer might handle CPU usage

    if _MODEL is None or model_name != _MODEL_NAME_LOADED or device != _DEVICE_LOADED:
        if torch is not None and device != "mps": # MPS device has its own seed handling
             # Attempt to set seed only if torch is available and not using MPS.
             # Some operations in SentenceTransformer might be non-deterministic otherwise.
            torch.manual_seed(0)

        try:
            # local_files_only=True can be problematic if model not pre-downloaded.
            # Consider making this configurable or False by default for easier use.
            # For now, keeping original logic.
            _MODEL = SentenceTransformer(
                model_name,
                device=device,
                # local_files_only=True, # Original had this, might be too restrictive
            )
        except Exception as exc: # pragma: no cover
            # Simplified error, original had compact-memory download-model suggestion
            raise RuntimeError(
                f"Error loading SentenceTransformer model '{model_name}' on device '{device}'. "
                f"Ensure the model is available or try a different model/device. Error: {exc}"
            ) from exc
        _MODEL_NAME_LOADED = model_name
        _DEVICE_LOADED = device
    if _MODEL is None: # Should not happen if previous block executed correctly
        raise RuntimeError("Model could not be loaded.")
    return _MODEL


def load_model(model_name: str = DEFAULT_MODEL_NAME, device: str = DEFAULT_DEVICE) -> None:
    """Explicitly load the Hugging Face embedding model."""
    _load_model(model_name, device)


def unload_model() -> None:
    """Unload the cached Hugging Face embedding model to free memory."""
    global _MODEL, _MODEL_NAME_LOADED, _DEVICE_LOADED
    _MODEL = None
    _MODEL_NAME_LOADED = None
    _DEVICE_LOADED = None
    # Attempt to clear GPU cache if torch is available
    if torch and hasattr(torch, 'cuda') and torch.cuda.is_available(): # pragma: no cover
        torch.cuda.empty_cache()


@contextlib.contextmanager
def loaded_embedding_model(
    model_name: str = DEFAULT_MODEL_NAME, device: str = DEFAULT_DEVICE
) -> SentenceTransformer: # Yields SentenceTransformer, not None
    """Context manager that loads and unloads the Hugging Face embedding model."""
    model = _load_model(model_name, device)
    try:
        yield model
    finally:
        # Decide if unload_model() is always called. If used in multiple places,
        # nested contexts might unload model prematurely.
        # For now, let's assume it's a top-level context or user manages nesting.
        unload_model()


@functools.lru_cache(maxsize=5000) # This decorator helps cache results for identical text inputs
def _embed_single_text_cached(
    text: str, model_name: str, device: str, batch_size: int # batch_size is for model.encode not direct here
) -> np.ndarray:
    """Internal cached function for embedding a single string."""
    model = _load_model(model_name, device)
    # model.encode can take a single string or list. For single string, batch_size is less relevant here.
    vec = model.encode(text, batch_size=batch_size, convert_to_numpy=True)
    vec = vec.astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0: # Avoid division by zero for zero vectors
        return vec
    vec /= norm
    return vec

def embed_text_hf(
    text: Union[str, Sequence[str]],
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = DEFAULT_DEVICE,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """
    Embeds a single text or a sequence of texts using a Hugging Face SentenceTransformer model.
    Vectors are normalized.
    """
    model = _load_model(model_name, device)

    # Text pre-processing (tiktoken length check, dynamic importance filter)
    # This was part of the original embed_text, applying it here per text.
    _tokenizer_for_length_check = None
    if tiktoken:
        try:
            _tokenizer_for_length_check = tiktoken.get_encoding("gpt2")
        except Exception: # pragma: no cover
            pass

    def _preprocess_text(t: str) -> str:
        is_too_long = False
        if _tokenizer_for_length_check:
            try:
                is_too_long = len(_tokenizer_for_length_check.encode(t)) > 1000 # Max token length
            except Exception: # pragma: no cover
                is_too_long = len(t.split()) > 1000 # Fallback
        else: # Fallback if tiktoken not available
            is_too_long = len(t.split()) > 1000

        if is_too_long:
            return dynamic_importance_filter(t)
        return t

    if isinstance(text, str):
        if not text: # Empty string
            return np.zeros(get_embedding_dim_hf(model_name, device), dtype=np.float32)
        processed_text = _preprocess_text(text)
        return _embed_single_text_cached(processed_text, model_name, device, batch_size)

    # Handle sequence of texts
    if not text: # Empty list/sequence
        return np.zeros((0, get_embedding_dim_hf(model_name, device)), dtype=np.float32)

    processed_texts = [_preprocess_text(t) for t in text]

    # Use model.encode directly for batch processing of sequences
    embeddings = model.encode(processed_texts, batch_size=batch_size, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Set zero norms to 1e-10 to avoid division by zero, effectively keeping zero vectors as zero
    norms[norms == 0] = 1e-10
    normalized_embeddings = embeddings / norms

    return normalized_embeddings


def get_embedding_dim_hf(model_name: str = DEFAULT_MODEL_NAME, device: str = DEFAULT_DEVICE) -> int:
    """Return the embedding dimension for the specified Hugging Face model."""
    model = _load_model(model_name, device)
    get_dim_func = getattr(model, "get_sentence_embedding_dimension", None)
    if callable(get_dim_func):
        dim = get_dim_func()
        if isinstance(dim, int):
            return dim
    # Fallback for some models that might store dimension in a 'dim' attribute
    dim_attr = getattr(model, "dim", None)
    if isinstance(dim_attr, int):
        return dim_attr

    # As a last resort, embed a dummy string and check its dimension
    try:
        dummy_embedding = model.encode("test", convert_to_numpy=True)
        return dummy_embedding.shape[-1]
    except Exception as e: # pragma: no cover
        raise AttributeError(
            f"Could not determine embedding dimension for model {model_name} on device {device}. Error: {e}"
        )


__all__ = [
    "load_model",
    "unload_model",
    "loaded_embedding_model",
    "embed_text_hf",
    "get_embedding_dim_hf",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_DEVICE",
    "DEFAULT_BATCH_SIZE",
]
