from __future__ import annotations

from typing import Dict, List, Protocol, Tuple, Optional

import numpy as np


class BaseVectorStore(Protocol):
    """Abstract vector store interface."""

    def add_vector(
        self, id: str, vector: np.ndarray, metadata: Optional[dict] = None
    ) -> None:
        pass

    def query_vector(self, query: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        pass

    def get_vector(self, id: str) -> np.ndarray:
        pass

    def delete_vector(self, id: str) -> None:
        pass

    def persist(self) -> None:
        """Persist to disk if supported."""
        pass

    def load(self) -> None:
        """Load from disk if supported."""
        pass


class InMemoryVectorStore:
    """Simple in-memory store using a Python dictionary."""

    def __init__(self, embedding_dim: int) -> None:
        self.embedding_dim = embedding_dim
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, dict] = {}

    def add_vector(
        self, id: str, vector: np.ndarray, metadata: Optional[dict] = None
    ) -> None:
        if vector.ndim != 1 or vector.shape[0] != self.embedding_dim:
            raise ValueError("vector dimension mismatch")
        norm = np.linalg.norm(vector) or 1.0
        self.vectors[id] = vector.astype(np.float32) / norm
        if metadata is not None:
            self.metadata[id] = metadata

    def query_vector(self, query: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        if query.ndim != 1:
            query = query.reshape(-1)
        norm = np.linalg.norm(query) or 1.0
        q = query.astype(np.float32) / norm
        sims = [(id, float(np.dot(q, vec))) for id, vec in self.vectors.items()]
        sims.sort(key=lambda x: -x[1])
        return sims[:top_k]

    def get_vector(self, id: str) -> np.ndarray:
        return self.vectors[id]

    def delete_vector(self, id: str) -> None:
        self.vectors.pop(id, None)
        self.metadata.pop(id, None)

    def persist(self) -> None:
        pass

    def load(self) -> None:
        pass
