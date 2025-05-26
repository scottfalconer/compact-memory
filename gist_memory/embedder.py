import numpy as np


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
