from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Type, Optional # Added Optional
import os
import json
import logging # Added
import numpy as np
import faiss
from datetime import datetime, timezone

from .models import BeliefPrototype, RawMemory
from .exceptions import VectorStoreError, IndexRebuildError # Added


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

    @abstractmethod
    def count(self) -> int:
        """Return the number of prototypes in the store."""

    @abstractmethod
    def get_texts_by_ids(self, ids: List[str]) -> Dict[str, str]:
        """Retrieve texts for a given list of prototype IDs."""

    @abstractmethod
    def add_texts_with_ids_and_vectors(self, data: List[Tuple[str, str, np.ndarray]]) -> None:
        """
        Add multiple entries, each consisting of an ID, text, and its vector.
        The text is primarily for association and retrieval via get_texts_by_ids.
        The vector is added to the searchable index, associated with the ID.
        """

    @abstractmethod
    def rebuild_index(self) -> None:
        """
        Forces a rebuild of the search index (e.g., FAISS index).
        Useful if the underlying data might have changed in a way not tracked by _index_dirty,
        or to explicitly reconstruct the index.
        """


class InMemoryVectorStore(VectorStore):
    """
    Simple in-memory vector store.

    This store keeps all data (vectors, texts, metadata) in memory.
    It now supports persistence by saving its state to disk, allowing it
    to be reloaded in a subsequent session. It is primarily used for
    testing and scenarios where a lightweight, non-persistent (by default)
    or file-based persistent store is sufficient.
    """

    def __init__(self, embedding_dim: int, normalized: bool = True) -> None:
        self.embedding_dim: int = embedding_dim
        self.normalized: bool = normalized
        self.meta: Dict[str, object] = {}
        self.path: Optional[str] = None # Typed self.path
        self.prototypes: List[BeliefPrototype] = []
        self.proto_vectors: np.ndarray = np.zeros((0, embedding_dim), dtype=np.float32) # Typed more generally
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
        # Ensure vec is 2D for vstack
        vec_2d = vec.reshape(1, -1) if vec.ndim == 1 else vec
        self.proto_vectors = np.vstack([self.proto_vectors, vec_2d]).astype(np.float32)
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

        if self.faiss_index is None: # Check if faiss_index is still None after build attempt (e.g. if no prototypes)
            return []

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

    def count(self) -> int:
        return len(self.prototypes)

    def get_texts_by_ids(self, ids: List[str]) -> Dict[str, str]:
        texts_map: Dict[str, str] = {}
        # Create a temporary lookup for faster access if many IDs are requested
        memory_lookup: Dict[str, str] = {mem.memory_id: mem.raw_text for mem in self.memories}
        for memory_id in ids:
            if memory_id in memory_lookup:
                texts_map[memory_id] = memory_lookup[memory_id]
        return texts_map

    def add_texts_with_ids_and_vectors(self, data: List[Tuple[str, str, np.ndarray]]) -> None:
        for entry_id, text, vector_np in data:
            # Assuming entry_id is used as prototype_id and memory_id
            # The BeliefPrototype might need more fields if they are used (e.g. created_at)
            # For now, vector_row_index is set by add_prototype
            proto = BeliefPrototype(prototype_id=entry_id, vector_row_index=0)

            # Calculate hash for RawMemory, assuming text is the primary content for hashing
            # This part might need alignment with how BaseCompressionEngine calculates hashes if consistency is key
            from compact_memory.utils import calculate_sha256 # Changed to absolute import
            text_hash = calculate_sha256(text)

            raw_memory = RawMemory(
                memory_id=entry_id,
                raw_text_hash=text_hash,
                raw_text=text,
                embedding=vector_np.tolist() # Storing vector in RawMemory, consistent with current model
            )

            self.add_prototype(proto, vector_np) # add_prototype handles setting vector_row_index
            self.add_memory(raw_memory)
        self._index_dirty = True # Ensure Faiss index is rebuilt if data was added

    def save(self, path: str) -> None:
        """
        Persist the entire state of the InMemoryVectorStore to the given path.

        Saves:
        - `embeddings.npy`: The numpy array of all vectors.
        - `text_entries.json`: A JSON file containing all RawMemory objects (text, original IDs, hashes).
        - `prototypes_meta.json`: A JSON file containing all BeliefPrototype objects (metadata about prototypes).

        Args:
            path: The directory path where the store's data will be saved.
                  The directory will be created if it doesn't exist.
        """
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "embeddings.npy"), self.proto_vectors)

        memories_to_save = [mem.model_dump(mode="json") for mem in self.memories]
        with open(os.path.join(path, "text_entries.json"), "w", encoding="utf-8") as fh:
            json.dump(memories_to_save, fh)

        prototypes_to_save = [p.model_dump(mode="json") for p in self.prototypes]
        with open(os.path.join(path, "prototypes_meta.json"), "w", encoding="utf-8") as fh:
            json.dump(prototypes_to_save, fh)
        logging.debug(f"InMemoryVectorStore state saved to {path}")

    def load(self, path: str) -> None:
        """
        Load the state of the InMemoryVectorStore from the given path.

        Restores:
        - Vectors from `embeddings.npy`.
        - RawMemory objects from `text_entries.json`.
        - BeliefPrototype objects from `prototypes_meta.json`.
        - Rebuilds the internal ID-to-index mapping.
        - Marks the FAISS index as dirty to be rebuilt on next access.

        Args:
            path: The directory path from which to load the store's data.
        """
        self.proto_vectors = np.load(os.path.join(path, "embeddings.npy"))

        with open(os.path.join(path, "text_entries.json"), "r", encoding="utf-8") as fh:
            memories_data = json.load(fh)
        self.memories = [RawMemory(**mem_data) for mem_data in memories_data]

        with open(os.path.join(path, "prototypes_meta.json"), "r", encoding="utf-8") as fh:
            prototypes_data = json.load(fh)
        self.prototypes = [BeliefPrototype(**p_data) for p_data in prototypes_data]

        # Rebuild index (mapping prototype_id to its integer index/row in self.prototypes and self.proto_vectors)
        self.index = {p.prototype_id: i for i, p in enumerate(self.prototypes)}

        # Mark Faiss index as dirty so it's rebuilt on next access
        self._index_dirty = True
        logging.debug(f"InMemoryVectorStore state loaded from {path}")

    def rebuild_index(self) -> None:
        """Forces a rebuild of the FAISS index."""
        logging.debug(f"InMemoryVectorStore: Rebuilding FAISS index.")
        self._build_faiss_index()
        logging.debug(f"InMemoryVectorStore: FAISS index rebuilt.")


class PersistentFaissVectorStore(InMemoryVectorStore):
    """FAISS-based store that persists data to disk."""

    def __init__(
        self, embedding_dim: int, normalized: bool = True, path: str | None = None
    ) -> None:
        super().__init__(embedding_dim, normalized)
        self.path = path

    def save(self, path: str | None = None) -> None:
        """
        Persist the entire state of the PersistentFaissVectorStore to the given path.

        Saves all data similar to InMemoryVectorStore (prototypes, vectors, memories)
        using its own filenames (`prototypes.json`, `vectors.npy`, `memories.json`),
        and additionally saves the FAISS index to `index.faiss`.

        Args:
            path: The directory path where the store's data will be saved.
                  If None, uses the path provided during initialization.
                  The directory will be created if it doesn't exist.

        Raises:
            VectorStoreError: If any error occurs during the save process.
        """
        actual_path = path or self.path
        if actual_path is None:
            raise ValueError("path must be provided for persistence")
        self.path = actual_path # Ensure self.path is updated if path was provided
        # TRY block should start here to encompass all operations
        try:
            os.makedirs(actual_path, exist_ok=True)
            with open(os.path.join(actual_path, "prototypes.json"), "w", encoding="utf-8") as fh:
                json.dump([p.model_dump(mode="json") for p in self.prototypes], fh)
            np.save(os.path.join(actual_path, "vectors.npy"), self.proto_vectors)
            with open(os.path.join(actual_path, "memories.json"), "w", encoding="utf-8") as fh:
                json.dump([m.model_dump(mode="json") for m in self.memories], fh)

            if self.faiss_index is None or self._index_dirty:
                self._build_faiss_index()

            if self.faiss_index is not None:
                faiss.write_index(self.faiss_index, os.path.join(actual_path, "index.faiss"))
            logging.info(f"PersistentFaissVectorStore saved to {actual_path}")
        except Exception as e:
            logging.error(f"Failed to save PersistentFaissVectorStore to {actual_path}: {e}")
            raise VectorStoreError(f"Failed to save PersistentFaissVectorStore to {actual_path}: {e}") from e


    def load(self, path: str | None = None) -> None:
        """
        Load the state of the PersistentFaissVectorStore from the given path.

        Restores data similar to InMemoryVectorStore (prototypes, vectors, memories)
        from its own filenames, and critically, loads the FAISS index from `index.faiss`.
        If the FAISS index file is not found, it attempts to rebuild it from the loaded vectors.

        Args:
            path: The directory path from which to load the store's data.
                  If None, uses the path provided during initialization.

        Raises:
            VectorStoreError: If any error occurs during loading, including
                              FileNotFoundError for essential files.
        """
        actual_path = path or self.path
        if actual_path is None:
            raise ValueError("path must be provided for loading")
        self.path = actual_path # Ensure self.path is updated
        # TRY block should start here
        try:
            with open(os.path.join(actual_path, "prototypes.json"), "r", encoding="utf-8") as fh:
                proto_data = json.load(fh)
            self.prototypes = [BeliefPrototype(**p) for p in proto_data]
            self.index = {p.prototype_id: i for i, p in enumerate(self.prototypes)}
            self.proto_vectors = np.load(os.path.join(actual_path, "vectors.npy"))
            with open(os.path.join(actual_path, "memories.json"), "r", encoding="utf-8") as fh:
                mem_data = json.load(fh)
            self.memories = [RawMemory(**m) for m in mem_data]

            index_file_path = os.path.join(actual_path, "index.faiss")
            if os.path.exists(index_file_path):
                self.faiss_index = faiss.read_index(index_file_path)
                self._index_dirty = False
            else:
                logging.warning(f"FAISS index file not found at {index_file_path}. Will attempt to rebuild.")
                self._build_faiss_index()
            logging.info(f"PersistentFaissVectorStore loaded from {actual_path}")
        except FileNotFoundError as e:
            logging.error(f"Required file not found during PersistentFaissVectorStore load from {actual_path}: {e}")
            raise VectorStoreError(f"File not found during load from {actual_path}: {e}") from e
        except Exception as e:
            logging.error(f"Failed to load PersistentFaissVectorStore from {actual_path}: {e}")
            raise VectorStoreError(f"Failed to load PersistentFaissVectorStore from {actual_path}: {e}") from e

    def rebuild_index(self) -> None:
        """
        Forces a rebuild of the FAISS index and persists it if a path is configured.

        Raises:
            IndexRebuildError: If persisting the rebuilt index fails.
        """
        logging.info(f"PersistentFaissVectorStore: Rebuilding FAISS index for path {self.path or 'in-memory'}.")
        self._build_faiss_index() # Rebuilds in-memory FAISS index
        logging.debug(f"PersistentFaissVectorStore: In-memory FAISS index rebuilt.")
        if self.path and self.faiss_index is not None:
            try:
                index_file_path = os.path.join(self.path, "index.faiss")
                faiss.write_index(self.faiss_index, index_file_path)
                logging.info(f"PersistentFaissVectorStore: Rebuilt FAISS index persisted to {index_file_path}")
            except Exception as e:
                logging.error(f"Failed to persist rebuilt FAISS index to {self.path}: {e}")
                raise IndexRebuildError(f"Failed to persist rebuilt FAISS index to {self.path}: {e}") from e
        elif self.path and self.faiss_index is None and len(self.prototypes) == 0:
            # If there are no prototypes, the index is None. If a path exists, try to remove the old index file.
            index_file_path = os.path.join(self.path, "index.faiss")
            if os.path.exists(index_file_path):
                try:
                    os.remove(index_file_path)
                    logging.info(f"PersistentFaissVectorStore: Removed stale FAISS index file as store is empty: {index_file_path}")
                except Exception as e:
                    logging.warning(f"PersistentFaissVectorStore: Failed to remove stale FAISS index file {index_file_path}: {e}")
        elif not self.path:
            logging.debug("PersistentFaissVectorStore: No path configured, rebuilt index not persisted externally.")


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

    # If the store type is 'in_memory', do not pass 'path' argument, even if it's in kwargs.
    if store_type == "in_memory":
        kwargs.pop("path", None) # Remove path if it exists, do nothing if not

    return cls(**kwargs)


__all__ = [
    "VectorStore",
    "InMemoryVectorStore",
    "PersistentFaissVectorStore",
    "create_vector_store",
]
