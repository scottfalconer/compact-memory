from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Type
import os
import json

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
    def save(self, path: str) -> None:
        """Persist the store to ``path`` (optional for in-memory stores)."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the store from ``path`` if applicable."""


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

    def save(self, path: str) -> None:  # pragma: no cover - nothing to persist
        pass

    def load(self, path: str) -> None:  # pragma: no cover - nothing to load
        pass


class PersistentFaissVectorStore(InMemoryVectorStore):
    """FAISS-based store that persists data to disk."""

    def __init__(
        self, embedding_dim: int, normalized: bool = True, path: str | None = None
    ) -> None:
        super().__init__(embedding_dim, normalized)
        self.path = path

    def save(self, path: str | None = None) -> None:
        path = path or self.path
        if path is None:
            raise ValueError("path must be provided for persistence")
        self.path = path
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "prototypes.json"), "w", encoding="utf-8") as fh:
            json.dump([p.model_dump(mode="json") for p in self.prototypes], fh)
        np.save(os.path.join(path, "vectors.npy"), self.proto_vectors)
        with open(os.path.join(path, "memories.json"), "w", encoding="utf-8") as fh:
            json.dump([m.model_dump(mode="json") for m in self.memories], fh)
        if self.faiss_index is None or self._index_dirty:
            self._build_faiss_index()

        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, os.path.join(path, "index.faiss"))

    def load(self, path: str | None = None) -> None:
        path = path or self.path
        if path is None:
            raise ValueError("path must be provided for loading")
        self.path = path
        with open(os.path.join(path, "prototypes.json"), "r", encoding="utf-8") as fh:
            proto_data = json.load(fh)
        self.prototypes = [BeliefPrototype(**p) for p in proto_data]
        self.index = {p.prototype_id: i for i, p in enumerate(self.prototypes)}
        self.proto_vectors = np.load(os.path.join(path, "vectors.npy"))
        with open(os.path.join(path, "memories.json"), "r", encoding="utf-8") as fh:
            mem_data = json.load(fh)
        self.memories = [RawMemory(**m) for m in mem_data]
        index_path = os.path.join(path, "index.faiss")
        if os.path.exists(index_path):
            self.faiss_index = faiss.read_index(index_path)
            self._index_dirty = False
        else:
            self._build_faiss_index()


_VECTOR_STORE_REGISTRY: Dict[str, Type[VectorStore]] = {
    "in_memory": InMemoryVectorStore,
    "faiss_persistent": PersistentFaissVectorStore,
}


def create_vector_store(store_type: str, **kwargs) -> VectorStore:
    """Instantiate a :class:`VectorStore` from ``store_type``."""

    if store_type not in _VECTOR_STORE_REGISTRY:
        known = ", ".join(sorted(_VECTOR_STORE_REGISTRY))
        raise ValueError(f"Unknown vector store '{store_type}'. Known types: {known}")
    cls = _VECTOR_STORE_REGISTRY[store_type]
    return cls(**kwargs)


__all__ = [
    "VectorStore",
    "InMemoryVectorStore",
    "PersistentFaissVectorStore",
    "create_vector_store",
]
