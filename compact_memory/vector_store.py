from __future__ import annotations

from typing import Dict, List, Tuple, Protocol, Optional
import numpy as np
import faiss

# Removed BeliefPrototype and RawMemory as they are no longer used in this file
# from .models import BeliefPrototype, RawMemory


class BaseVectorStore(Protocol):
    """Protocol for vector stores used by Compact Memory."""

    def add_vector(self, id: str, vector: np.ndarray, metadata: Optional[Dict] = None) -> None:
        """Adds a vector to the store."""
        ...

    def query_vector(self, query: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """Queries the store for the top_k most similar vectors."""
        ...

    def get_vector(self, id:str) -> np.ndarray:
        """Retrieves a vector by its ID."""
        ...

    def delete_vector(self, id: str) -> None:
        """Deletes a vector by its ID."""
        ...

    def persist(self) -> None:
        """Persists the vector store to disk. (Optional)"""
        ...

    def load(self) -> None:
        """Loads the vector store from disk. (Optional)"""
        ...


class InMemoryVectorStore:
    """Simple in-memory implementation of BaseVectorStore."""

    def __init__(self, embedding_dim: int, path: str | None = None) -> None:
        self.embedding_dim: int = embedding_dim
        self.path: Optional[str] = path  # Not used for now

        self.vectors: Dict[str, np.ndarray] = {}
        self.faiss_id_to_internal_id: List[str] = []

        self.faiss_index: Optional[faiss.Index] = None
        self._index_dirty: bool = True # Faiss index needs rebuild

    def _rebuild_faiss_index(self) -> None:
        if not self._index_dirty:
            return

        if not self.vectors:
            self.faiss_index = None
            self.faiss_id_to_internal_id = []
            self._index_dirty = False
            return

        # Rebuild the faiss_id_to_internal_id list from current vectors
        self.faiss_id_to_internal_id = list(self.vectors.keys())

        temp_vectors = np.array([self.vectors[id] for id in self.faiss_id_to_internal_id]).astype(np.float32)

        if temp_vectors.shape[0] == 0: # Should be caught by `if not self.vectors` but as a safeguard
            self.faiss_index = None
            self._index_dirty = False
            return

        # Normalize vectors before adding to FAISS
        norms = np.linalg.norm(temp_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10 # Avoid division by zero
        normalized_vectors = temp_vectors / norms

        index = faiss.IndexFlatIP(self.embedding_dim) # Using Inner Product for similarity
        index.add(normalized_vectors)
        self.faiss_index = index
        self._index_dirty = False

    def add_vector(self, id: str, vector: np.ndarray, metadata: Optional[Dict] = None) -> None:
        if id in self.vectors:
            # Overwriting existing vector, mark index dirty
            # For simplicity, we can just update the vector and let rebuild handle FAISS.
            # More advanced: update in FAISS if supported and vector dim matches.
            pass

        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        if vector.shape[1] != self.embedding_dim:
            raise ValueError(f"Vector dimension {vector.shape[1]} does not match embedding_dim {self.embedding_dim}")

        self.vectors[id] = vector.flatten() # Store as 1D array
        # Instead of directly manipulating faiss_id_to_internal_id here,
        # we mark the index as dirty and let _rebuild_faiss_index handle it.
        self._index_dirty = True
        # metadata is ignored in this implementation

    def query_vector(self, query: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        if self._index_dirty or self.faiss_index is None:
            self._rebuild_faiss_index()

        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return []

        if query.ndim == 1:
            query = query.reshape(1, -1)

        if query.shape[1] != self.embedding_dim:
            raise ValueError(f"Query vector dimension {query.shape[1]} does not match embedding_dim {self.embedding_dim}")

        # Normalize query vector
        norm = np.linalg.norm(query)
        if norm == 0: # Avoid division by zero for zero vectors
            normalized_query = query.astype(np.float32)
        else:
            normalized_query = (query / norm).astype(np.float32)

        distances, indices = self.faiss_index.search(normalized_query, min(top_k, self.faiss_index.ntotal))

        results: List[Tuple[str, float]] = []
        for i, dist in zip(indices[0], distances[0]):
            if i < 0: # Should not happen with IndexFlatIP unless k > ntotal
                continue
            internal_id = self.faiss_id_to_internal_id[i]
            results.append((internal_id, float(dist)))
        return results

    def get_vector(self, id: str) -> np.ndarray:
        if id not in self.vectors:
            raise KeyError(f"Vector with id '{id}' not found.")
        return self.vectors[id]

    def delete_vector(self, id: str) -> None:
        if id not in self.vectors:
            # Optionally raise an error or just return if non-existent
            return

        del self.vectors[id]
        # Mark index as dirty. _rebuild_faiss_index will regenerate faiss_id_to_internal_id
        self._index_dirty = True

    def persist(self) -> None:
        # No-op for in-memory store
        pass

    def load(self) -> None:
        # No-op for in-memory store
        pass


__all__ = ["BaseVectorStore", "InMemoryVectorStore"]
