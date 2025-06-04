from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import BeliefPrototype, RawMemory
from .vector_store import BaseVectorStore, InMemoryVectorStore


class MemoryStore:
    """Ephemeral memory store using a vector backend."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        embedding_dim: int,
        vector_store: BaseVectorStore | None = None,
    ) -> None:
        self.path = Path(path) if path is not None else None
        if self.path is not None:
            os.makedirs(self.path, exist_ok=True)
        self.embedding_dim = embedding_dim
        self.vector_store = vector_store or InMemoryVectorStore(embedding_dim)
        self.prototypes: List[BeliefPrototype] = []
        self.memories: List[RawMemory] = []
        self.index: Dict[str, int] = {}
        self.meta: Dict[str, object] = {"embedding_dim": embedding_dim}

    # --------------------------------------------------------------
    def add_prototype(self, proto: BeliefPrototype, vec: np.ndarray) -> None:
        idx = len(self.prototypes)
        proto.vector_row_index = idx
        self.prototypes.append(proto)
        self.index[proto.prototype_id] = idx
        self.vector_store.add_vector(proto.prototype_id, vec)

    # --------------------------------------------------------------
    def update_prototype(
        self,
        proto_id: str,
        new_vec: np.ndarray,
        memory_id: str,
        *,
        alpha: float | None = None,
    ) -> float:
        idx = self.index[proto_id]
        current = self.vector_store.get_vector(proto_id)
        proto = self.prototypes[idx]
        if alpha is None:
            alpha = 1.0 / (proto.strength + 1.0) if proto.strength > 0 else 1.0
        updated = (1 - alpha) * current + alpha * new_vec
        change = float(np.linalg.norm(updated - current))
        norm = np.linalg.norm(updated) or 1.0
        updated = updated / norm
        self.vector_store.add_vector(proto_id, updated)
        proto.last_updated_ts = datetime.now(timezone.utc).replace(microsecond=0)
        proto.constituent_memory_ids.append(memory_id)
        proto.strength += 1.0
        return change

    # --------------------------------------------------------------
    def find_nearest(self, vec: np.ndarray, k: int) -> List[Tuple[str, float]]:
        return self.vector_store.query_vector(vec, k)

    # --------------------------------------------------------------
    def add_memory(self, memory: RawMemory) -> None:
        self.memories.append(memory)

    # --------------------------------------------------------------
    def save(self) -> None:
        # No-op for in-memory store
        pass

    def load(self) -> None:
        # No-op for in-memory store
        pass
