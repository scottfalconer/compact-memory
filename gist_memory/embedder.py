from __future__ import annotations

import numpy as np
import openai

try:  # optional dependency used for local embeddings
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - library may not be installed during tests
    SentenceTransformer = None


class Embedder:
    """Base embedding interface."""

    def embed(self, text: str) -> np.ndarray:
        raise NotImplementedError()


class RandomEmbedder(Embedder):
    """Generate a random embedding. Placeholder for real model."""

    def __init__(self, dim: int = 768, seed: int | None = None):
        self.dim = dim
        self.rng = np.random.default_rng(seed)

    def embed(self, text: str) -> np.ndarray:
        return self.rng.random(self.dim)


class OpenAIEmbedder(Embedder):
    """Embed text using the OpenAI API."""

    def __init__(self, model: str = "text-embedding-ada-002"):
        self.model = model

    def embed(self, text: str) -> np.ndarray:
        resp = openai.Embedding.create(input=[text], model=self.model)
        data = resp["data"][0]["embedding"]
        return np.array(data, dtype=np.float32)


class LocalEmbedder(Embedder):
    """Embed text using a locally runnable SentenceTransformer model."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        *,
        local_files_only: bool = True,
    ):
        """Create a local embedder.

        Parameters
        ----------
        model_name:
            Name or path of the model to load.
        local_files_only:
            If ``True`` (the default), disable any network calls and only use
            locally cached model files. This allows the embedder to function in
            fully offline environments.
        """

        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required for LocalEmbedder"
            )
        self.model = SentenceTransformer(
            model_name,
            local_files_only=local_files_only,
        )

    def embed(self, text: str) -> np.ndarray:
        vec = self.model.encode(text)
        return np.array(vec, dtype=np.float32)


def get_embedder(kind: str = "random", model_name: str | None = None) -> Embedder:
    """Utility to create an embedder by name."""

    if kind == "openai":
        return OpenAIEmbedder(model=model_name or "text-embedding-ada-002")
    if kind == "local":
        return LocalEmbedder(model_name=model_name or "all-MiniLM-L6-v2")
    return RandomEmbedder()
