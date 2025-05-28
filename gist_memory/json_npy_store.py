from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

from .models import BeliefPrototype, RawMemory
from .embedding_pipeline import EmbeddingDimensionMismatchError


class VectorStore:
    """Abstract storage interface."""

    def add_prototype(self, proto: BeliefPrototype, vec: np.ndarray) -> None:
        raise NotImplementedError

    def update_prototype(self, proto_id: str, new_vec: np.ndarray, memory_id: str) -> None:
        raise NotImplementedError

    def find_nearest(self, vec: np.ndarray, k: int) -> List[Tuple[str, float]]:
        raise NotImplementedError

    def add_memory(self, memory: RawMemory) -> None:
        raise NotImplementedError

    def save(self) -> None:
        raise NotImplementedError

    def load(self) -> None:
        raise NotImplementedError


class JsonNpyVectorStore(VectorStore):
    """Filesystem-backed store using JSON + NPY files."""

    def __init__(self, path: str, *, embedding_model: str = "unknown", embedding_dim: int = 0, normalized: bool = True) -> None:
        self.path = Path(path)
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.normalized = normalized
        self.meta: Dict[str, object] = {}
        self.prototypes: List[BeliefPrototype] = []
        self.proto_vectors: np.ndarray | None = None
        self.memories: List[RawMemory] = []
        self.index: Dict[str, int] = {}
        if self.path.exists() and (self.path / "meta.yaml").exists():
            self.load()
        else:
            os.makedirs(self.path, exist_ok=True)
            self.meta = {
                "version": 1,
                "embedding_model": self.embedding_model,
                "embedding_dim": self.embedding_dim,
                "normalized": self.normalized,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "updated_at": datetime.utcnow().isoformat() + "Z",
            }
            self.proto_vectors = np.zeros((0, self.embedding_dim), dtype=np.float32)
            self.save()

    # ------------------------------------------------------------------
    def _meta_path(self) -> Path:
        return self.path / "meta.yaml"

    def _proto_json_path(self) -> Path:
        return self.path / "belief_prototypes.json"

    def _proto_vec_path(self) -> Path:
        return self.path / "prototype_vectors.npy"

    def _mem_jsonl_path(self) -> Path:
        return self.path / "raw_memories.jsonl"

    # ------------------------------------------------------------------
    def load(self) -> None:
        if not self._meta_path().exists():
            raise FileNotFoundError("meta.yaml missing")
        self.meta = yaml.safe_load(self._meta_path().read_text())
        self.embedding_model = str(self.meta.get("embedding_model", "unknown"))
        self.embedding_dim = int(self.meta.get("embedding_dim", 0))
        self.normalized = bool(self.meta.get("normalized", False))
        if not self.normalized:
            raise ValueError("embeddings must be normalized")

        if self._proto_vec_path().exists():
            arr = np.load(self._proto_vec_path())
            if arr.ndim != 2 or arr.shape[1] != self.embedding_dim:
                raise EmbeddingDimensionMismatchError(
                    f"{arr.shape[1]} vs {self.embedding_dim}"
                )
            self.proto_vectors = arr.astype(np.float32)
        else:
            self.proto_vectors = np.zeros((0, self.embedding_dim), dtype=np.float32)

        if self._proto_json_path().exists():
            with open(self._proto_json_path(), "r") as f:
                data = json.load(f)
            self.prototypes = [BeliefPrototype(**p) for p in data]
            self.index = {p.prototype_id: p.vector_row_index for p in self.prototypes}
        if self._mem_jsonl_path().exists():
            with open(self._mem_jsonl_path(), "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    self.memories.append(RawMemory.parse_raw(line))

    def save(self) -> None:
        self.meta["updated_at"] = datetime.utcnow().isoformat() + "Z"
        with open(self._meta_path(), "w") as f:
            yaml.safe_dump(self.meta, f)
        with open(self._proto_json_path(), "w") as f:
            json.dump([p.model_dump(mode="json") for p in self.prototypes], f)
        if self.proto_vectors is not None:
            np.save(self._proto_vec_path(), self.proto_vectors)
        with open(self._mem_jsonl_path(), "w") as f:
            for mem in self.memories:
                f.write(mem.model_dump_json() + "\n")

    # ------------------------------------------------------------------
    def add_prototype(self, proto: BeliefPrototype, vec: np.ndarray) -> None:
        idx = len(self.prototypes)
        proto.vector_row_index = idx
        self.prototypes.append(proto)
        if self.proto_vectors is None:
            self.proto_vectors = vec.reshape(1, -1)
        else:
            self.proto_vectors = np.vstack([self.proto_vectors, vec])
        self.index[proto.prototype_id] = idx

    def update_prototype(self, proto_id: str, new_vec: np.ndarray, memory_id: str) -> None:
        idx = self.index[proto_id]
        assert self.proto_vectors is not None
        self.proto_vectors[idx] = new_vec
        proto = self.prototypes[idx]
        proto.last_updated_ts = datetime.utcnow().replace(microsecond=0)
        proto.constituent_memory_ids.append(memory_id)

    def find_nearest(self, vec: np.ndarray, k: int) -> List[Tuple[str, float]]:
        if self.proto_vectors is None or len(self.prototypes) == 0:
            return []
        dists = np.linalg.norm(self.proto_vectors - vec, axis=1)
        idxs = np.argsort(dists)[:k]
        return [(self.prototypes[i].prototype_id, float(dists[i])) for i in idxs]

    def add_memory(self, memory: RawMemory) -> None:
        self.memories.append(memory)


