from __future__ import annotations

from typing import Dict, List, Tuple, Protocol
import numpy as np
import faiss

from .models import BeliefPrototype, RawMemory


class BaseVectorStore(Protocol):
    """Protocol for vector stores used by Compact Memory."""

    prototypes: List[BeliefPrototype]
    memories: List[RawMemory]
    meta: Dict[str, object]

    def add_prototype(self, proto: BeliefPrototype, vec: np.ndarray) -> None:
        pass

    def update_prototype(
        self,
        proto_id: str,
        new_vec: np.ndarray,
        memory_id: str,
        *,
        alpha: float | None = None,
    ) -> float:
        pass

    def find_nearest(self, vec: np.ndarray, k: int) -> List[Tuple[str, float]]:
        pass

    def add_memory(self, memory: RawMemory) -> None:
        pass

    def save(self) -> None:
        pass

    def load(self) -> None:
        pass


class InMemoryVectorStore:
    """Simple in-memory implementation for testing and demos."""

    def __init__(self, embedding_dim: int, path: str | None = None) -> None:
        self.embedding_dim = embedding_dim
        self.path = path
        self.prototypes: List[BeliefPrototype] = []
        self.proto_vectors: np.ndarray | None = None
        self.memories: List[RawMemory] = []
        self.index: Dict[str, int] = {}
        self.faiss_index: faiss.Index | None = None
        self.meta: Dict[str, object] = {"embedding_dim": embedding_dim}
        self._index_dirty = True

    # ------------------------------------------------------------------
    def _build_faiss_index(self) -> None:
        if self.proto_vectors is None or len(self.prototypes) == 0:
            self.faiss_index = None
            self._index_dirty = False
            return
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(self.proto_vectors.astype(np.float32))
        self.faiss_index = index
        self._index_dirty = False

    # ------------------------------------------------------------------
    def add_prototype(self, proto: BeliefPrototype, vec: np.ndarray) -> None:
        idx = len(self.prototypes)
        proto.vector_row_index = idx
        self.prototypes.append(proto)
        if self.proto_vectors is None:
            self.proto_vectors = vec.reshape(1, -1)
        else:
            if vec.ndim == 1:
                vec = vec.reshape(1, -1)
            self.proto_vectors = np.vstack([self.proto_vectors, vec])
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
        assert self.proto_vectors is not None
        current = self.proto_vectors[idx]
        proto = self.prototypes[idx]
        if alpha is None:
            alpha = 1.0 / (proto.strength + 1.0) if proto.strength > 0 else 1.0
        updated = (1 - alpha) * current + alpha * new_vec
        change = float(np.linalg.norm(updated - current))
        norm = np.linalg.norm(updated) or 1.0
        updated = updated / norm
        self.proto_vectors[idx] = updated.astype(np.float32)
        proto.last_updated_ts = proto.last_updated_ts
        proto.constituent_memory_ids.append(memory_id)
        proto.strength += 1.0
        self._index_dirty = True
        return change

    def find_nearest(self, vec: np.ndarray, k: int) -> List[Tuple[str, float]]:
        if self.proto_vectors is None or len(self.prototypes) == 0:
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

    def save(self) -> None:
        # No-op for in-memory store
        pass

    def load(self) -> None:
        # No-op for in-memory store
        pass


__all__ = ["BaseVectorStore", "InMemoryVectorStore"]
