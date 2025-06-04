from __future__ import annotations

from typing import Sequence, Union, Callable, Optional # Added Callable, Optional
import numpy as np

# Imports for moved HF functionality are now in embedding_providers.huggingface
# We will import the new default functions from there.
from .embedding_providers.huggingface import (
    embed_text_hf,
    get_embedding_dim_hf,
    DEFAULT_MODEL_NAME as HF_DEFAULT_MODEL_NAME, # Avoid name clashes if any
    DEFAULT_DEVICE as HF_DEFAULT_DEVICE,
    DEFAULT_BATCH_SIZE as HF_DEFAULT_BATCH_SIZE
)
# Keep MockEncoder and EmbeddingDimensionMismatchError if they are general
# MockEncoder was used for testing, so it can stay.
import hashlib # For MockEncoder

class EmbeddingDimensionMismatchError(ValueError):
    """Raised when stored embedding dimension does not match model output."""

class MockEncoder:
    """Deterministic mock encoder used in tests."""
    dim = 32

    def encode(self, texts: Union[str, Sequence[str]], *args, **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        arr = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hashlib.sha256(t.encode("utf-8")).digest()
            # Ensure consistent slicing for the mock encoder
            arr[i] = np.frombuffer(h[:self.dim], dtype=np.uint8) / 255.0

        # Original MockEncoder returned arr[0] if len(arr) == 1, which is a 1D array.
        # The EmbeddingFunction type expects np.ndarray (which can be 1D or 2D).
        # For consistency with embed_text_hf (which returns 2D for single string if batched, or 1D if single),
        # it's better if mock also returns 1D for single string, and 2D for list.
        # However, SentenceTransformer.encode typically returns 2D for list of 1, and 1D for single string.
        # Let's make MockEncoder return 1D for single string, 2D for list of strings.
        if len(texts) == 1 and isinstance(texts, list) and len(texts[0]) > 0 : # A list containing one string
             # This case is tricky. If a list of one string is passed, ST returns 2D array.
             # If a single string is passed, ST returns 1D array.
             # The type hint is Union[str, Sequence[str]].
             # Let's align with what embed_text_hf will do for single string vs list.
             # The internal _embed_single_text_cached returns 1D.
             # The embed_text_hf for a single string input eventually calls _embed_single_text_cached.
             # For a list of strings, it calls model.encode which for HF returns 2D.
             pass # arr is already (len(texts), self.dim)

        if isinstance(texts, str): # Single string input
            return arr[0]
        return arr # Sequence input


# Define the EmbeddingFunction type alias
EmbeddingFunction = Callable[[Union[str, Sequence[str]]], np.ndarray]

# Global/default model parameters are now managed within huggingface.py for HF provider.
# This pipeline can still define defaults if it were to choose between providers,
# but for now, it defaults to HF.

def embed_text(
    text: Union[str, Sequence[str]],
    embedding_fn: Optional[EmbeddingFunction] = None,
    *,
    # Parameters for the default Hugging Face embedder, if embedding_fn is None
    model_name: str = HF_DEFAULT_MODEL_NAME,
    device: str = HF_DEFAULT_DEVICE,
    batch_size: int = HF_DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """
    Embeds text using a provided embedding function or defaults to Hugging Face.

    Args:
        text: The text or sequence of texts to embed.
        embedding_fn: An optional custom function to use for embedding.
                      If None, the default Hugging Face SentenceTransformer is used.
        model_name: Model name for the default Hugging Face embedder.
        device: Device for the default Hugging Face embedder.
        batch_size: Batch size for the default Hugging Face embedder.

    Returns:
        A NumPy array containing the embedding(s).
        If input is a single string, output is a 1D array.
        If input is a sequence of strings, output is a 2D array (num_texts, embedding_dim).
    """
    if embedding_fn:
        return embedding_fn(text)
    else:
        # Default to Hugging Face provider
        return embed_text_hf(
            text,
            model_name=model_name,
            device=device,
            batch_size=batch_size
        )

def get_embedding_dim(
    # No longer takes embedding_fn, as dimension cannot be reliably determined from it.
    # Users of custom embedding_fn must provide embedding_dim separately.
    model_name: str = HF_DEFAULT_MODEL_NAME,
    device: str = HF_DEFAULT_DEVICE
) -> int:
    """
    Returns the embedding dimension for the default Hugging Face model.

    For custom embedding functions, the embedding dimension must be known
    and provided separately when configuring components like Agent or Vector Stores.
    """
    # Delegates to the Hugging Face specific dimension getter
    return get_embedding_dim_hf(model_name=model_name, device=device)


# register_embedding is removed as it's less relevant with EmbeddingFunction being passable directly.
# Model loading/unloading functions are now specific to HF provider in huggingface.py.
# Users can manage lifecycle of custom embedding_fn externally.

__all__ = [
    "EmbeddingDimensionMismatchError",
    "MockEncoder",
    "EmbeddingFunction", # Export the type alias
    "embed_text",
    "get_embedding_dim",
]
