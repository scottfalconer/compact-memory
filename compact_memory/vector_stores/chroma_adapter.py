from __future__ import annotations

from typing import List, Tuple, Dict, Optional, Any
import numpy as np

from compact_memory.vector_store import BaseVectorStore

try:
    import chromadb
    from chromadb.api import ClientAPI as ChromaClientAPI # Specific type
    from chromadb.api.models.collections import Collection as ChromaCollection
    # We don't strictly need EmbeddingFunction type from chromadb for hints if we use Any
    # or if we pass it through. If a specific EF type is needed for get_or_create_collection,
    # it might be 'Any' or a more specific type if available and safe to import.
    # from chromadb.api.types import EmbeddingFunction as ChromaEmbeddingFunction
    CHROMADB_INSTALLED = True
except ImportError:
    chromadb = None  # Define chromadb as None to avoid NameError at runtime

    # Define placeholder types for type hinting if chromadb is not installed
    # These allow the class to be defined and imported even if chromadb is missing.
    class ChromaClientAPI:  # type: ignore
        def get_or_create_collection(self, name: str, embedding_function: Optional[Any] = None, metadata: Optional[Dict] = None) -> "ChromaCollection": # type: ignore
            pass
        def __getattr__(self, name: str) -> Any: # Basic mock
            if name in ("PersistentClient", "EphemeralClient"): return lambda *args, **kwargs: self
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    class ChromaCollection:  # type: ignore
        def add(self, ids: List[str], embeddings: List[List[float]], metadatas: Optional[List[Optional[Dict]]] = None) -> None:
            pass
        def query(self, query_embeddings: List[List[float]], n_results: int) -> Dict[str, Optional[List[Any]]]:
            return {}
        def get(self, ids: List[str], include: List[str]) -> Dict[str, Optional[List[Any]]]:
            return {'ids': [], 'embeddings': None, 'metadatas': None, 'documents': None, 'uris': None, 'data': None}
        def delete(self, ids: List[str]) -> None:
            pass
        def __getattr__(self, name: str) -> Any: # Basic mock
             raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


    # class ChromaEmbeddingFunction: pass # type: ignore

    CHROMADB_INSTALLED = False

DEFAULT_COLLECTION_NAME = "compact_memory_default"

class ChromaVectorStoreAdapter(BaseVectorStore):
    """
    Vector store adapter for ChromaDB.
    Manages pre-computed vectors.
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        path: Optional[str] = None,
        client: Optional[ChromaClientAPI] = None,
        # Use Any for embedding_function for broader compatibility without chromadb installed
        # If CHROMADB_INSTALLED, this could be Optional[chromadb.EmbeddingFunction]
        embedding_function: Optional[Any] = None,
        collection_metadata: Optional[Dict] = None,
    ):
        """
        Initializes the ChromaVectorStoreAdapter.

        Args:
            collection_name: Name of the collection to use or create.
            path: Path for ChromaDB PersistentClient. If None, EphemeralClient is used.
            client: An existing chromadb.ClientAPI instance.
            embedding_function: An optional embedding function for the collection.
                                Note: This adapter primarily deals with pre-computed vectors.
                                If an embedding function is provided, it's passed to Chroma,
                                but `add_vector` expects already computed embeddings.
                                Defaults to None, letting Chroma use its own default if it has one,
                                or operate without if embeddings are always provided.
            collection_metadata: Optional metadata for the collection.
        """
        if not CHROMADB_INSTALLED:
            # Option 1: Raise an error
            # raise ImportError("ChromaDB is not installed. Please install it to use ChromaVectorStoreAdapter.")
            # Option 2: Log a warning and proceed (methods will fail if called)
            print("Warning: ChromaDB not found. ChromaVectorStoreAdapter will not function.")
            # Option 3: Set a flag and check in each method (makes methods more complex)
            self._functional = False
        else:
            self._functional = True


        self.collection_name = collection_name
        self._client: ChromaClientAPI
        if client:
            self._client = client
        elif chromadb: # Check chromadb module is not None
            if path:
                self._client = chromadb.PersistentClient(path=path)
            else:
                self._client = chromadb.EphemeralClient()
        elif not self._functional: # chromadb is None and we are not functional
             # Create a placeholder client if not functional to avoid NoneErrors on self._client
             # This client's methods will not work, but it allows object instantiation.
            self._client = ChromaClientAPI() # Placeholder
        else:
            # This case should ideally not be reached if CHROMADB_INSTALLED is true
            # and client is None, as chromadb should be available.
            # However, as a fallback:
            print("Warning: ChromaDB client could not be initialized despite CHROMADB_INSTALLED being true.")
            self._client = ChromaClientAPI() # Placeholder
            self._functional = False


        if self._functional and chromadb:
            # The collection's embedding function is tricky. BaseVectorStore provides raw vectors.
            # If an EF is set here, Chroma might expect text for .add().
            # For now, we pass it, but assume .add() with embeddings bypasses it.
            # A safer default for our use case (pre-computed vectors) is to pass ef=None
            # unless the user explicitly provides one for other reasons.
            # Chroma's default EF is SentenceTransformer if ef=None and no existing collection metadata has other.
            # To ensure we store precomputed vectors, it's often best to use `ef=None` or a dummy EF
            # if the collection doesn't already exist with a different EF.

            # If an embedding_function is explicitly passed, use it. Otherwise, use None.
            # This makes it more explicit that we are dealing with pre-computed embeddings primarily.
            final_embedding_function = embedding_function

            self.collection: ChromaCollection = self._client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=final_embedding_function, # Pass user's EF or None
                metadata=collection_metadata
            )
        else:
            # If not functional, use a placeholder collection
            self.collection = ChromaCollection()


    def add_vector(self, id: str, vector: np.ndarray, metadata: Optional[Dict] = None) -> None:
        if not self._functional:
            # print("Error: ChromaDB not installed or client not functional.")
            raise RuntimeError("ChromaDB adapter is not functional. Check ChromaDB installation.")

        if vector.ndim == 1:
            embedding_list = vector.tolist()
        elif vector.ndim == 2 and vector.shape[0] == 1:
            embedding_list = vector.flatten().tolist()
        else:
            raise ValueError("Input vector must be 1D or 2D with one row.")

        metadatas_list = [metadata] if metadata else None
        try:
            self.collection.add(ids=[id], embeddings=[embedding_list], metadatas=metadatas_list)
        except Exception as e:
            # Catching generic Exception from Chroma, can be refined if specific exceptions are known
            print(f"Error adding vector to Chroma: {e}")
            # Optionally re-raise or handle
            raise

    def query_vector(self, query: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        if not self._functional:
            raise RuntimeError("ChromaDB adapter is not functional. Check ChromaDB installation.")

        if query.ndim == 1:
            query_list = query.tolist()
        elif query.ndim == 2 and query.shape[0] == 1:
            query_list = query.flatten().tolist()
        else:
            raise ValueError("Query vector must be 1D or 2D with one row.")

        try:
            results = self.collection.query(query_embeddings=[query_list], n_results=top_k)
        except Exception as e:
            print(f"Error querying Chroma: {e}")
            # Optionally re-raise or handle
            raise

        output: List[Tuple[str, float]] = []
        if results and results.get('ids') and results.get('distances'):
            ids_list = results['ids'][0]  # query returns lists of lists
            distances_list = results['distances'][0]

            if len(ids_list) != len(distances_list):
                # This shouldn't happen with a well-behaved Chroma client
                print("Warning: Mismatch in length of IDs and distances from Chroma query.")
                return []

            for item_id, dist in zip(ids_list, distances_list):
                output.append((str(item_id), float(dist)))
        return output

    def get_vector(self, id: str) -> np.ndarray:
        if not self._functional:
            raise RuntimeError("ChromaDB adapter is not functional. Check ChromaDB installation.")
        try:
            result = self.collection.get(ids=[id], include=['embeddings'])
        except Exception as e:
            print(f"Error getting vector from Chroma: {e}")
            raise

        if result and result.get('embeddings') and result['embeddings'] and len(result['embeddings'][0]) > 0:
            # Chroma returns list of embeddings, get the first one
            embedding = result['embeddings'][0]
            return np.array(embedding, dtype=np.float32)
        else:
            raise KeyError(f"Vector with id '{id}' not found in Chroma collection.")

    def delete_vector(self, id: str) -> None:
        if not self._functional:
            raise RuntimeError("ChromaDB adapter is not functional. Check ChromaDB installation.")
        try:
            self.collection.delete(ids=[id])
        except Exception as e:
            print(f"Error deleting vector from Chroma: {e}")
            raise

    def persist(self) -> None:
        """
        ChromaDB PersistentClient handles persistence automatically.
        This method is a no-op.
        """
        if not self._functional:
            # Allow no-op even if not functional, as it doesn't do anything.
            pass
        # If self._client is a PersistentClient, persist might be available,
        # but typically it's managed by ChromaDB itself.
        # e.g., if hasattr(self._client, 'persist'): self._client.persist()
        pass

    def load(self) -> None:
        """
        ChromaDB loads data on client initialization.
        This method is a no-op.
        """
        if not self._functional:
            pass
        pass

__all__ = ["ChromaVectorStoreAdapter"]
