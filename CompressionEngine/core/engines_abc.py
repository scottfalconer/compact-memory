from __future__ import annotations

"""Abstract interface for memory compression engines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Union, Any, Optional, Tuple

from .trace import CompressionTrace


@dataclass
class CompressedMemory:
    """
    Simple container for compressed memory text and associated metadata.

    Attributes:
        text: The compressed string output from a compression engine.
        metadata: An optional dictionary to hold any additional information
                  about the compressed content, such as source IDs, timestamps,
                  or engine-specific details.
    """

    text: str
    metadata: Optional[Dict[str, Any]] = None


class CompressionEngine(ABC):
    """
    Abstract Base Class for defining memory compression engines.

    Engine developers should subclass this class and implement the `compress` method.
    Each engine must also have a unique `id` class attribute, which is a string
    used by the framework to identify and register the engine.

    Example:
    ```python
    from CompressionEngine.core.engines_abc import (
        CompressionEngine,
        CompressedMemory,
        CompressionTrace,
    )

    class MyCustomEngine(CompressionEngine):
        id = "my_custom_engine"

        def compress(self, text_or_chunks, llm_token_budget, **kwargs):
            # ... implementation ...
            compressed_text = "..."
            trace_details = CompressionTrace(...)
            return CompressedMemory(text=compressed_text), trace_details
    ```

    Optionally, engines can implement `save_learnable_components` and
    `load_learnable_components` if they involve trainable models or state
    that needs to be persisted and reloaded.
    """

    id: str  # Unique string identifier for the engine.

    @abstractmethod
    def compress(
        self,
        text_or_chunks: Union[str, List[str]],
        llm_token_budget: int,
        **kwargs: Any,
    ) -> Tuple[CompressedMemory, CompressionTrace]:
        """
        Compresses the input text or list of chunks to meet the LLM token budget.

        This method must be implemented by concrete engine subclasses.

        Args:
            text_or_chunks: The input text to be compressed. This can be a single
                            string or a list of strings (e.g., pre-chunked text).
            llm_token_budget: The target maximum number of tokens (or a similar unit
                              like characters, depending on the engine's internal logic)
                              that the compressed output should ideally have. The engine
                              should strive to keep the output within this budget.
            **kwargs: Additional keyword arguments that specific engines might require
                      or that the calling framework might provide. Common examples include:
                      - `tokenizer`: A tokenizer instance (e.g., from Hugging Face) that
                        can be used for accurate token counting and text processing.
                      - `metadata`: Any other relevant metadata that might influence the
                        compression process.

        Returns:
            A tuple containing:
                - CompressedMemory: An object holding the `text` attribute with the
                                  compressed string result, and optionally `metadata`.
                - CompressionTrace: An object logging details about the compression
                                  process (e.g., engine name, parameters, input/output
                                  token counts, steps taken, processing time). This is crucial
                                  for debugging, analysis, and understanding engine behavior.
        """

    def save_learnable_components(
        self, path: str
    ) -> None:  # pragma: no cover - optional
        """Saves any learnable components of the engine to the specified path."""

    def load_learnable_components(
        self, path: str
    ) -> None:  # pragma: no cover - optional
        """Loads any learnable components of the engine from the specified path."""


import uuid
from pathlib import Path
import json # For saving/loading JSON data
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime # Added for timestamp
from typing import Dict, List, Union, Any, Optional, Tuple, Type

from compact_memory.models import RawMemory # Assuming RawMemory is suitable
# Ensure other necessary imports are here or add them:
from compact_memory.chunkers import Chunker, SentenceWindowChunker
from compact_memory.embedding_pipeline import embed_text_sync
from compact_memory.vector_store import BaseVectorStore, InMemoryVectorStore
from .config import BaseEngineConfig # Import the new base config
from .trace import CompressionTrace # Already imported


@dataclass
class QueryResult:
    """Represents a single result from a recall operation."""
    memory: RawMemory
    score: float


class BaseCompressionEngine(ABC):
    """
    Base class for compression engines, providing common functionalities
    for ingestion, recall, and persistence.
    """
    id: str = "base_engine" # Default ID, should be overridden by subclasses
    display_name: Optional[str] = "Base Engine" # Default display name
    config_class: Optional[Type[BaseEngineConfig]] = BaseEngineConfig

    def __init__(self, config: Optional[BaseEngineConfig] = None, memory_path: Optional[str] = None):
        self.config = config if config is not None else (self.config_class() if self.config_class else None)
        self.memory_path: Optional[Path] = Path(memory_path) if memory_path else None

        # Initialize store - default to InMemoryVectorStore.
        # Specific engines can override this in their __init__ or via config.
        # For now, embedding_dim is hardcoded; ideally, it comes from config or embedding model.
        # This will be more robustly handled when config loading for engines is refined.
        embedding_dim = 384 # Default for all-MiniLM-L6-v2, replace with dynamic getter
        try:
            # Attempt to get embedding dimension dynamically if possible
            # This is a placeholder for a more robust mechanism.
            from compact_memory.embedding_pipeline import get_embedding_model_dim_from_name
            embedding_dim = get_embedding_model_dim_from_name(self._get_embedding_model_name())
        except Exception:
            pass # Use hardcoded default if dynamic fetch fails

        self.store: Optional[BaseVectorStore] = InMemoryVectorStore(embedding_dim=embedding_dim)

        self.memories: Dict[str, RawMemory] = {} # Store by ID for easier access
        self.chunker: Chunker = SentenceWindowChunker() # Default chunker

        # Ensure embedding model is available (conceptual check)
        # Actual loading/checking might occur in embed_texts or a dedicated setup method.
        _ = self._get_embedding_model_name()

    def ingest(self, text: str, metadata: Optional[Dict[str, Any]] = None, trace_scope: Optional[str] = None, **kwargs) -> List[str]:
        """
        Chunks text, compresses each chunk, embeds it, and stores it.

        Args:
            text: The text to ingest.
            metadata: Optional metadata to associate with the ingested memories.
            trace_scope: Optional scope for tracing this specific ingestion.
            **kwargs: Additional arguments for compression or storage.

        Returns:
            A list of IDs of the ingested RawMemory objects.
        """
        if trace_scope: # Basic way to pass trace info, could be more structured
            # This is a placeholder, actual tracing integration needs to be more robust
            pass

        chunk_texts = self.chunker.chunk(text)
        ingested_memory_ids: List[str] = []

        for i, chunk_text_content in enumerate(chunk_texts):
            # Allow metadata to be customized per chunk if needed, or use shared metadata
            chunk_metadata = (metadata or {}).copy()
            chunk_metadata["chunk_index"] = i
            # original_text is the uncompressed chunk text for this specific chunk
            compressed_chunk = self.compress_chunk(chunk_text_content, original_text=chunk_text_content, **chunk_metadata)

            embedding_array = self.embed_texts([compressed_chunk.text])
            if embedding_array.ndim == 0 or embedding_array.size == 0: # Check if embedding is empty
                # Decide how to handle empty embeddings: skip, log, error?
                # For now, let's assume we might skip or use a zero vector if appropriate.
                # This example will use a zero vector of the correct dimension.
                embedding = np.zeros(self.store.embedding_dim if self.store else 384, dtype=np.float32) # Fallback
            else:
                embedding = embedding_array[0]


            memory_id = str(uuid.uuid4())
            # Ensure RawMemory can store original_text if different from compressed_chunk.text after compression
            raw_memory = RawMemory(
                id=memory_id, # Assuming RawMemory has an 'id' field
                text=compressed_chunk.text,
                original_text=chunk_text_content, # Store original pre-compressed chunk
                embedding=embedding.tolist(), # Store as list
                metadata=compressed_chunk.metadata,
                timestamp=datetime.utcnow().isoformat() # Example timestamp
            )
            self.memories[memory_id] = raw_memory
            ingested_memory_ids.append(memory_id)

            if self.store:
                self.store.add_entry(memory_id, embedding, metadata=raw_memory.metadata)

        return ingested_memory_ids

    def recall(self, query: str, k: int = 5, **kwargs) -> List[QueryResult]:
        """
        Embeds a query and retrieves the top k most similar memories.
        """
        if not self.store:
            return []

        query_embedding = self.embed_texts(query)
        if query_embedding.ndim == 0 or query_embedding.size == 0:
            return [] # Cannot recall if query embedding failed

        # find_nearest returns List[Tuple[str, float]] (id, score)
        search_results = self.store.find_nearest(query_embedding[0], k=k)

        query_results: List[QueryResult] = []
        for memory_id, score in search_results:
            memory = self.memories.get(memory_id)
            if memory:
                query_results.append(QueryResult(memory=memory, score=score))
        return query_results

    def compress_chunk(self, chunk_text: str, trace: Optional[CompressionTrace] = None, **kwargs) -> CompressedMemory:
        """
        Compresses a single chunk of text. Base implementation is identity.
        kwargs may include metadata to be attached to the CompressedMemory.
        """
        # original_text is passed via kwargs if needed by a specific engine, or could be chunk_text
        original_text = kwargs.pop("original_text", chunk_text)
        return CompressedMemory(text=chunk_text, metadata={"original_text": original_text, **kwargs})

    def compress(self, text: str, budget: Optional[int] = None, trace: Optional[CompressionTrace] = None, **kwargs) -> CompressedMemory:
        """
        Performs one-shot compression of a larger text.
        Base implementation calls compress_chunk and ignores budget.
        Subclasses should override for more sophisticated budgeting and compression.
        """
        # Pass along budget and other kwargs if compress_chunk is made more aware
        return self.compress_chunk(text, trace=trace, original_text=text, **kwargs)

    def _get_embedding_model_name(self) -> str:
        """
        Returns the name of the embedding model to be used.
        Can be overridden by subclasses or configured via self.config.
        """
        # Example: if self.config and hasattr(self.config, "embedding_model_name"):
        # return self.config.embedding_model_name
        return "all-MiniLM-L6-v2" # Default

    def embed_texts(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        Embeds one or more texts using the configured embedding model.
        """
        if isinstance(texts, str):
            texts = [texts]
        if not texts: # Return empty array if no texts to embed
            return np.array([], dtype=np.float32).reshape(0, self.store.embedding_dim if self.store and hasattr(self.store, 'embedding_dim') else 384)


        # Using actual_embed_text which was correctly identified before
        from compact_memory.embedding_pipeline import embed_text as actual_embed_text
        embeddings_list = actual_embed_text(texts, model_name=self._get_embedding_model_name(), normalize_embeddings=normalize)

        # Handle cases where embedding might fail for some texts or return None
        processed_embeddings = []
        default_embedding_dim = self.store.embedding_dim if self.store and hasattr(self.store, 'embedding_dim') else 384
        for emb in embeddings_list:
            if emb is None:
                # Use a zero vector if embedding failed for a specific text
                processed_embeddings.append(np.zeros(default_embedding_dim, dtype=np.float32))
            else:
                processed_embeddings.append(np.array(emb, dtype=np.float32))

        if not processed_embeddings: # If all embeddings failed, return empty array of correct shape
             return np.array([], dtype=np.float32).reshape(0, default_embedding_dim)

        return np.stack(processed_embeddings)

    def get_statistics(self) -> Dict[str, Any]:
        """Returns basic statistics about the engine's state."""
        return {
            "memory_count": len(self.memories),
            "vector_store_entries": self.store.get_entry_count() if self.store else 0,
            "embedding_model": self._get_embedding_model_name(),
            "config": self.config.to_dict() if self.config else None,
        }

    def save(self, path: Optional[str] = None) -> None:
        """
        Saves the engine's state to the specified path.
        Includes engine manifest, memories, and vector store data.
        """
        save_path = Path(path or self.memory_path or ".")
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)
        elif not save_path.is_dir():
            raise ValueError(f"Save path '{save_path}' must be a directory.")

        # Save engine manifest
        manifest = {
            "engine_id": self.id,
            "engine_class": self.__class__.__module__ + "." + self.__class__.__name__,
            "config": self.config.to_dict() if self.config else None,
            "embedding_model": self._get_embedding_model_name(),
            "memory_count": len(self.memories),
        }
        with open(save_path / "engine_manifest.json", "w") as f:
            json.dump(manifest, f, indent=4)

        # Save memories (excluding embeddings for separate optimized storage)
        memories_to_save = []
        memory_ids_in_order = []
        embeddings_list = []

        for mem_id, memory_obj in self.memories.items():
            # Create a dict from RawMemory, but without the embedding field for memories.json
            mem_dict = {f.name: getattr(memory_obj, f.name) for f in dataclasses.fields(RawMemory) if f.name != 'embedding'} # Use RawMemory here
            memories_to_save.append(mem_dict)

            # Store embeddings and their corresponding IDs for ordered saving
            if memory_obj.embedding: # Ensure there is an embedding
                memory_ids_in_order.append(mem_id)
                embeddings_list.append(memory_obj.embedding)

        with open(save_path / "memories.jsonl", "w") as f:
            for mem_data in memories_to_save:
                f.write(json.dumps(mem_data) + "\n")

        if embeddings_list:
            embeddings_array = np.array(embeddings_list, dtype=np.float32)
            np.save(save_path / "vectors.npy", embeddings_array)
            with open(save_path / "vector_ids.json", "w") as f:
                json.dump(memory_ids_in_order, f)

        # If self.store has its own optimized save method (e.g., for FAISS index)
        if self.store and hasattr(self.store, 'save_index') and callable(self.store.save_index):
            self.store.save_index(str(save_path / "vector_store_index"))
        elif not embeddings_list and self.store and self.store.get_entry_count() > 0:
            # This case implies embeddings might be managed solely by the store internally
            # and not directly available via self.memories[id].embedding.
            # This part needs a strategy for stores that don't expose embeddings easily.
            # For now, we assume InMemoryVectorStore is repopulated from vectors.npy or this is handled by engine-specific save.
            print(f"Warning: Vector store for engine '{self.id}' may have entries not saved if embeddings are not in self.memories.")


    @classmethod
    def load(cls, path: str, config_override: Optional[BaseEngineConfig] = None) -> 'BaseCompressionEngine':
        """
        Loads an engine's state from the specified path.
        """
        load_path = Path(path)
        if not load_path.is_dir():
            raise FileNotFoundError(f"Load path '{load_path}' is not a valid directory.")

        with open(load_path / "engine_manifest.json", "r") as f:
            manifest = json.load(f)

        engine_id_from_manifest = manifest.get("engine_id")
        # engine_class_str = manifest.get("engine_class") # Resolving this dynamically is complex for now

        # Use provided config_override or load from manifest or default for the class
        final_config: Optional[BaseEngineConfig] = None
        if config_override:
            final_config = config_override
        elif manifest.get("config"):
            # Assuming config_class is correctly set on the class `cls`
            # For BaseCompressionEngine, this would be BaseEngineConfig
            config_data = manifest["config"]
            if hasattr(cls.config_class, "from_dict"): # Ideal scenario
                 final_config = cls.config_class.from_dict(config_data)
            elif cls.config_class: # Basic dataclass
                 final_config = cls.config_class(**config_data)

        # Instantiate the engine - uses memory_path from load_path
        # Specific engine might override __init__ to handle its specific store loading
        engine = cls(config=final_config, memory_path=str(load_path))
        engine.id = engine_id_from_manifest # Ensure ID from manifest is used

        # Load memories
        memories_file = load_path / "memories.jsonl"
        if memories_file.exists():
            with open(memories_file, "r") as f:
                for line in f:
                    mem_data = json.loads(line)
                    # Reconstruct RawMemory, potentially handling missing 'embedding' if it's stored separately
                    # Assuming RawMemory can be created from dict:
                    # Need to ensure RawMemory's __init__ or a factory can handle this dict
                    # For now, let's assume RawMemory can be directly instantiated.
                    # This part is tricky due to embedding potentially not being in mem_data.
                    # We'll populate embeddings after loading vectors.npy
                    raw_mem_fields = {f.name for f in dataclasses.fields(RawMemory)}
                    filtered_mem_data = {k: v for k, v in mem_data.items() if k in raw_mem_fields}
                    if 'id' not in filtered_mem_data and 'memory_id' in filtered_mem_data: # Adapt to potential id field name
                        filtered_mem_data['id'] = filtered_mem_data.pop('memory_id')

                    # If RawMemory expects embedding and it's not there, this will fail.
                    # We will add embeddings later.
                    if 'embedding' in filtered_mem_data: del filtered_mem_data['embedding']

                    # Handle timestamp (assuming it's ISO format string)
                    if 'timestamp' in filtered_mem_data and isinstance(filtered_mem_data['timestamp'], str):
                        # Placeholder: actual datetime object might be needed by RawMemory
                        pass # RawMemory's responsibility to parse if needed

                    memory_obj = RawMemory(**filtered_mem_data)
                    engine.memories[memory_obj.id] = memory_obj


        # Load vectors and rebuild store
        vectors_file = load_path / "vectors.npy"
        vector_ids_file = load_path / "vector_ids.json"

        if hasattr(engine.store, 'load_index') and callable(engine.store.load_index) and (load_path / "vector_store_index").exists():
            engine.store.load_index(str(load_path / "vector_store_index"))
        elif vectors_file.exists() and vector_ids_file.exists():
            vectors = np.load(vectors_file)
            with open(vector_ids_file, "r") as f:
                vector_ids = json.load(f)

            if not engine.store: # Should have been initialized in __init__
                 embedding_dim_from_vectors = vectors.shape[1] if vectors.ndim == 2 and vectors.shape[0] > 0 else 384
                 engine.store = InMemoryVectorStore(embedding_dim=embedding_dim_from_vectors)

            for mem_id, vector in zip(vector_ids, vectors):
                if mem_id in engine.memories: # Assign embedding back to RawMemory
                    engine.memories[mem_id].embedding = vector.tolist()
                # Add to store, assuming store can handle metadata if needed
                engine.store.add_entry(mem_id, vector, metadata=engine.memories[mem_id].metadata if mem_id in engine.memories else {})

        return engine

    @abstractmethod
    def save_learnable_components(self, path: str) -> None:
        pass

    @abstractmethod
    def load_learnable_components(self, path: str) -> None:
        pass


__all__ = ["CompressedMemory", "CompressionEngine", "CompressionTrace", "BaseCompressionEngine", "QueryResult"]
