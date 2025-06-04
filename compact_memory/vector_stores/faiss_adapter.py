from __future__ import annotations

import os
import pickle
from typing import List, Tuple, Dict, Optional, Any
import numpy as np

from compact_memory.vector_store import BaseVectorStore

try:
    import faiss
    FAISS_INSTALLED = True
except ImportError:
    faiss = None  # Define faiss as None to avoid NameError at runtime

    class Index:  # Placeholder for faiss.Index
        def __init__(self, d: int, metric: Any = None): pass # metric is for IndexFlatIP compatibility
        def add(self, vectors: np.ndarray) -> None: pass
        def reset(self) -> None: pass
        def search(self, query_vectors: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
            return (np.array([[]]), np.array([[]]))
        @property
        def ntotal(self) -> int: return 0
        @property
        def d(self) -> int: return 0 # dimension

    FAISS_INSTALLED = False

# Define constants for filenames for persistence
VECTORS_FILENAME = "vectors.pkl"
IDS_FILENAME = "ids.pkl"
FAISS_INDEX_FILENAME = "index.faiss"

class FaissVectorStoreAdapter(BaseVectorStore):
    """
    Vector store adapter for FAISS.
    Manages pre-computed vectors with a FAISS index for querying.
    Supports persistence to disk.
    """

    def __init__(
        self,
        embedding_dim: int,
        faiss_index: Optional[Any] = None, # faiss.Index | Index (placeholder)
        initial_vectors: Optional[Dict[str, np.ndarray]] = None,
        index_factory_string: str = "Flat", # e.g., "Flat", "IVF4096,Flat"
    ):
        if not FAISS_INSTALLED:
            raise ImportError("FAISS is not installed. Please install it to use FaissVectorStoreAdapter.")

        self.embedding_dim: int = embedding_dim
        self.vectors: Dict[str, np.ndarray] = {}
        self.faiss_id_to_internal_id: List[str] = []
        self._index_dirty: bool = True # Start dirty if new or initial vectors

        if faiss_index:
            if not hasattr(faiss_index, 'd') or faiss_index.d != self.embedding_dim:
                raise ValueError(
                    f"Provided FAISS index dimension ({faiss_index.d if hasattr(faiss_index, 'd') else 'unknown'}) "
                    f"does not match embedding_dim ({self.embedding_dim})"
                )
            self.faiss_index = faiss_index
            self._index_dirty = True # Assume dirty to sync with internal storages
        else:
            # Using faiss.METRIC_INNER_PRODUCT for similarity (cosine similarity on normalized vectors)
            self.faiss_index = faiss.index_factory(
                self.embedding_dim, index_factory_string, faiss.METRIC_INNER_PRODUCT
            )

        if initial_vectors:
            for id_str, vector in initial_vectors.items():
                if vector.shape[-1] != self.embedding_dim:
                    raise ValueError(
                        f"Initial vector for ID '{id_str}' has dimension {vector.shape[-1]}, "
                        f"expected {self.embedding_dim}"
                    )
                self.vectors[id_str] = vector.flatten() # Ensure 1D
            # self._rebuild_faiss_index_from_vectors() # Call rebuild after adding all
            self._index_dirty = True # Mark dirty, will be rebuilt on first query/add or explicitly

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalizes rows of a 2D array."""
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10  # Avoid division by zero
        return vectors / norms

    def _rebuild_faiss_index_from_vectors(self) -> None:
        if not FAISS_INSTALLED or not faiss: # Should not happen if __init__ guard works
            return

        self.faiss_index.reset() # Clear the FAISS index
        self.faiss_id_to_internal_id = [] # Clear the ID mapping

        if not self.vectors:
            self._index_dirty = False
            return

        internal_ids_list = list(self.vectors.keys())
        # Ensure vector_data is 2D
        vector_data = np.array([self.vectors[id_] for id_ in internal_ids_list], dtype=np.float32)
        if vector_data.ndim == 1: # Single vector case
            vector_data = vector_data.reshape(1, -1)

        if vector_data.shape[0] == 0: # No vectors to add
             self._index_dirty = False
             return

        normalized_vectors = self._normalize_vectors(vector_data)
        self.faiss_index.add(normalized_vectors)
        self.faiss_id_to_internal_id = internal_ids_list
        self._index_dirty = False

    def add_vector(self, id: str, vector: np.ndarray, metadata: Optional[Dict] = None) -> None:
        if vector.shape[-1] != self.embedding_dim:
            raise ValueError(f"Vector dimension {vector.shape[-1]} does not match store's embedding_dim {self.embedding_dim}")

        self.vectors[id] = vector.flatten() # Ensure 1D
        self._index_dirty = True
        # Metadata is ignored in this adapter

    def query_vector(self, query: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        if not FAISS_INSTALLED or not faiss:
            raise RuntimeError("FAISS not installed.")

        if self._index_dirty:
            self._rebuild_faiss_index_from_vectors()

        if self.faiss_index.ntotal == 0:
            return []

        if query.ndim == 1:
            query_vector_2d = query.reshape(1, -1)
        elif query.ndim == 2 and query.shape[0] == 1:
            query_vector_2d = query
        else:
            raise ValueError("Query vector must be 1D or 2D with one row.")

        if query_vector_2d.shape[-1] != self.embedding_dim:
             raise ValueError(f"Query vector dimension {query_vector_2d.shape[-1]} does not match store's embedding_dim {self.embedding_dim}")


        normalized_query = self._normalize_vectors(query_vector_2d.astype(np.float32))

        distances, indices = self.faiss_index.search(normalized_query, min(top_k, self.faiss_index.ntotal))

        results: List[Tuple[str, float]] = []
        for i in range(indices.shape[1]):
            faiss_idx = indices[0, i]
            if faiss_idx < 0: # Should not happen with non-exhaustive search unless k > ntotal
                continue
            dist = distances[0, i]
            internal_id = self.faiss_id_to_internal_id[faiss_idx]
            results.append((internal_id, float(dist)))
        return results

    def get_vector(self, id: str) -> np.ndarray:
        if id not in self.vectors:
            raise KeyError(f"Vector with id '{id}' not found.")
        return self.vectors[id]

    def delete_vector(self, id: str) -> None:
        if id in self.vectors:
            del self.vectors[id]
            self._index_dirty = True
        # If ID not found, do nothing or raise error? Current: do nothing.

    def persist(self, dir_path: str) -> None:
        if not FAISS_INSTALLED or not faiss:
            raise RuntimeError("FAISS not installed, cannot persist.")

        os.makedirs(dir_path, exist_ok=True)

        vectors_path = os.path.join(dir_path, VECTORS_FILENAME)
        with open(vectors_path, "wb") as f:
            pickle.dump(self.vectors, f)

        ids_path = os.path.join(dir_path, IDS_FILENAME)
        with open(ids_path, "wb") as f:
            pickle.dump(self.faiss_id_to_internal_id, f)

        # If index was dirty, rebuild before saving to ensure consistency
        if self._index_dirty:
            self._rebuild_faiss_index_from_vectors()

        faiss_index_path = os.path.join(dir_path, FAISS_INDEX_FILENAME)
        faiss.write_index(self.faiss_index, faiss_index_path)

        # Save embedding_dim for loading integrity
        config_path = os.path.join(dir_path, "config.pkl")
        with open(config_path, "wb") as f:
            pickle.dump({"embedding_dim": self.embedding_dim,
                         "index_factory_string": self.faiss_index.factory_string if hasattr(self.faiss_index, 'factory_string') else "Flat"}, f)


    @classmethod
    def load(cls, dir_path: str) -> "FaissVectorStoreAdapter":
        if not FAISS_INSTALLED or not faiss:
            raise ImportError("FAISS is not installed, cannot load.")

        config_path = os.path.join(dir_path, "config.pkl")
        with open(config_path, "rb") as f:
            config = pickle.load(f)
        embedding_dim = config["embedding_dim"]
        # index_factory_string = config.get("index_factory_string", "Flat") # For future use if needed

        vectors_path = os.path.join(dir_path, VECTORS_FILENAME)
        with open(vectors_path, "rb") as f:
            vectors = pickle.load(f)

        ids_path = os.path.join(dir_path, IDS_FILENAME)
        with open(ids_path, "rb") as f:
            faiss_id_to_internal_id = pickle.load(f)

        faiss_index_path = os.path.join(dir_path, FAISS_INDEX_FILENAME)
        loaded_faiss_index = faiss.read_index(faiss_index_path)

        if loaded_faiss_index.d != embedding_dim:
            raise ValueError(f"Loaded FAISS index dimension ({loaded_faiss_index.d}) "
                             f"does not match configured embedding_dim ({embedding_dim})")

        # Create an instance without calling __init__ logic directly, then populate
        instance = cls(embedding_dim=embedding_dim, faiss_index=loaded_faiss_index)
        instance.vectors = vectors
        instance.faiss_id_to_internal_id = faiss_id_to_internal_id
        instance._index_dirty = False # Loaded state is considered clean

        # Verify consistency
        if loaded_faiss_index.ntotal != len(faiss_id_to_internal_id):
            print(f"Warning: FAISS index size ({loaded_faiss_index.ntotal}) does not match "
                  f"loaded ID list size ({len(faiss_id_to_internal_id)}). Index might need rebuild.")
            instance._index_dirty = True


        return instance

__all__ = ["FaissVectorStoreAdapter"]
