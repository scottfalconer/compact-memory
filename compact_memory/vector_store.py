from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

import numpy as np
import faiss
from datetime import datetime, timezone

from .models import BeliefPrototype, RawMemory


class VectorStore(ABC):
    """Abstract interface for vector stores."""

    @abstractmethod
    def add_prototype(self, proto: BeliefPrototype, vec: np.ndarray) -> None:
        """Add a prototype and its vector to the store."""

    @abstractmethod
    def update_prototype(
        self,
        proto_id: str,
        new_vec: np.ndarray,
        memory_id: str,
        *,
        alpha: float | None = None,
    ) -> float:
        """Update ``proto_id`` towards ``new_vec`` and return magnitude of change."""

    @abstractmethod
    def find_nearest(self, vec: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Return ``k`` nearest prototype IDs and their similarity scores."""

    @abstractmethod
    def add_memory(self, memory: RawMemory) -> None:
        """Add a memory entry to the store."""

    @abstractmethod
    def save(self) -> None:
        """Persist the store (optional for in-memory stores)."""

    @abstractmethod
    def load(self) -> None:
        """Load the store from persistence if applicable."""


class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store used mainly for testing."""

    def __init__(self, embedding_dim: int, normalized: bool = True) -> None:
        self.embedding_dim = embedding_dim
        self.normalized = normalized
        self.meta: Dict[str, object] = {}
        self.path = None
        self.prototypes: List[BeliefPrototype] = []
        self.proto_vectors = np.zeros((0, embedding_dim), dtype=np.float32)
        self.memories: List[RawMemory] = []
        self.index: Dict[str, int] = {}
        self.faiss_index: faiss.Index | None = None
        self._index_dirty = True

    # --------------------------------------------------------------
    def _build_faiss_index(self) -> None:
        if len(self.prototypes) == 0:
            self.faiss_index = None
            self._index_dirty = False
            return
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(self.proto_vectors.astype(np.float32))
        self.faiss_index = index
        self._index_dirty = False

    # --------------------------------------------------------------
    def add_prototype(self, proto: BeliefPrototype, vec: np.ndarray) -> None:
        idx = len(self.prototypes)
        proto.vector_row_index = idx
        self.prototypes.append(proto)
        vec = vec.reshape(1, -1) if vec.ndim == 1 else vec
        self.proto_vectors = np.vstack([self.proto_vectors, vec]).astype(np.float32)
        self.index[proto.prototype_id] = idx
        self._index_dirty = True

    def update_prototype(
        self,
        proto_id: str,
        new_vec: np.ndarray,
        memory_id: str,
        *,
        alpha: float | None = None,
    ) -> float:
        idx = self.index[proto_id]
        current = self.proto_vectors[idx]
        proto = self.prototypes[idx]
        if alpha is None:
            alpha = 1.0 / (proto.strength + 1.0) if proto.strength > 0 else 1.0
        updated = (1 - alpha) * current + alpha * new_vec
        change = float(np.linalg.norm(updated - current))
        if self.normalized:
            norm = np.linalg.norm(updated) or 1.0
            updated = updated / norm
        self.proto_vectors[idx] = updated.astype(np.float32)
        proto.last_updated_ts = datetime.now(timezone.utc).replace(microsecond=0)
        proto.constituent_memory_ids.append(memory_id)
        proto.strength += 1.0
        self._index_dirty = True
        return change

    def find_nearest(self, vec: np.ndarray, k: int) -> List[Tuple[str, float]]:
        if len(self.prototypes) == 0:
            return []
        if self.faiss_index is None or self._index_dirty:
            self._build_faiss_index()
        query = vec.astype(np.float32).reshape(1, -1)
        dists, idxs = self.faiss_index.search(query, min(k, self.faiss_index.ntotal))
        results: List[Tuple[str, float]] = []
        for idx, dist in zip(idxs[0], dists[0]):
            if idx < 0:
                continue
            results.append((self.prototypes[int(idx)].prototype_id, float(dist)))
        return results

    def add_memory(self, memory: RawMemory) -> None:
        self.memories.append(memory)

    def save(self) -> None:  # pragma: no cover - nothing to persist
        pass

    def load(self) -> None:  # pragma: no cover - nothing to load
        pass
