from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set
import json
import os
import uuid
import inspect

import numpy as np


from ..chunker import Chunker, SentenceWindowChunker
from ..embedding_pipeline import embed_text, get_embedding_dim
from ..utils import calculate_sha256
from ..models import BeliefPrototype, RawMemory
from ..vector_store import InMemoryVectorStore, VectorStore


@dataclass
class CompressedMemory:
    """Container for compressed text and metadata."""

    text: str
    metadata: Optional[Dict[str, Any]] = None
    engine_id: Optional[str] = None
    engine_config: Optional[Dict[str, Any]] = None
    trace: Optional[CompressionTrace] = None


@dataclass
class CompressionTrace:
    """Trace metadata for a compression operation."""

    engine_name: str
    strategy_params: Dict[str, Any]
    input_summary: Dict[str, Any]
    steps: List[Dict[str, Any]] = field(default_factory=list)
    output_summary: Dict[str, Any] = field(default_factory=dict)
    processing_ms: float | None = None
    final_compressed_object_preview: Optional[str] = None


class BaseCompressionEngine:
    """Simple retrieval engine using MiniLM embeddings and FAISS."""

    id = "base"

    def __init__(
        self,
        *,
        chunker: Chunker | None = None,
        embedding_fn: Callable[[str | Sequence[str]], np.ndarray] = embed_text,
        preprocess_fn: Callable[[str], str] | None = None,
        vector_store: VectorStore | None = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,  # Allow other config params to be passed
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = kwargs  # Store kwargs if no explicit config dict is given

        # Prioritize chunker_id from config if it exists.
        # Otherwise, if a chunker instance is passed, derive its ID.
        # Otherwise, set a default chunker_id.
        if "chunker_id" not in self.config:
            if chunker:
                self.config["chunker_id"] = type(chunker).__name__
            else:
                self.config["chunker_id"] = type(SentenceWindowChunker()).__name__
        # elif chunker and type(chunker).__name__ != self.config['chunker_id']:
        # This case could be a warning or error if a chunker instance is passed
        # that doesn't match a pre-existing chunker_id in the config.
        # For now, we assume config['chunker_id'] (if present) is the source of truth.

        # Note: embedding_fn and preprocess_fn are not easily serializable by default.
        # Subclasses would need to handle serialization/deserialization if they are configurable.

        self._chunker = chunker or SentenceWindowChunker()
        self.embedding_fn = embedding_fn or embed_text
        self.preprocess_fn = preprocess_fn
        sig = inspect.signature(self.embedding_fn)
        self._embed_accepts_preprocess = "preprocess_fn" in sig.parameters
        self.memories: List[Dict[str, Any]] = []
        dim = get_embedding_dim()
        self.vector_store: VectorStore = vector_store or InMemoryVectorStore(dim)
        self.memory_hashes: Set[str] = set()

    # --------------------------------------------------
    def _compress_chunk(self, chunk_text: str) -> str:
        """
        Compresses a single text chunk.

        This method is intended to be overridden by subclasses to implement
        specific compression logic. By default, it returns the chunk unmodified.

        Args:
            chunk_text: The text chunk to compress.

        Returns:
            The compressed text chunk.
        """
        return chunk_text

    # --------------------------------------------------
    def _ensure_index(self) -> None:
        """Placeholder for backward compatibility."""
        # VectorStore implementations handle indexing internally.
        pass

    # --------------------------------------------------
    def _embed(self, text_or_texts: str | Sequence[str]) -> np.ndarray:
        """Return embeddings for ``text_or_texts``."""
        if self._embed_accepts_preprocess:
            return self.embedding_fn(text_or_texts, preprocess_fn=self.preprocess_fn)
        if self.preprocess_fn is not None:
            if isinstance(text_or_texts, str):
                text_or_texts = self.preprocess_fn(text_or_texts)
            else:
                text_or_texts = [self.preprocess_fn(t) for t in text_or_texts]
        return self.embedding_fn(text_or_texts)

    # --------------------------------------------------
    def ingest(self, text: str) -> List[str]:
        """Chunk and store ``text`` in the engine."""

        raw_chunks = self.chunker.chunk(text)
        if not raw_chunks:
            return []

        processed_chunks = [self._compress_chunk(chunk) for chunk in raw_chunks]

        vecs = self._embed(processed_chunks)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        ids: List[str] = []
        for processed_chunk_text, vec in zip(processed_chunks, vecs):
            chunk_hash = calculate_sha256(processed_chunk_text)
            if chunk_hash not in self.memory_hashes:
                mid = uuid.uuid4().hex
                self.memories.append({"id": mid, "text": processed_chunk_text})
                proto = BeliefPrototype(prototype_id=mid, vector_row_index=0)
                self.vector_store.add_prototype(proto, vec.astype(np.float32))
                memory = RawMemory(
                    memory_id=mid,
                    raw_text_hash=chunk_hash,
                    raw_text=processed_chunk_text,
                    embedding=vec.astype(np.float32).tolist(),
                )
                self.vector_store.add_memory(memory)
                ids.append(mid)
                self.memory_hashes.add(chunk_hash)
        return ids

    # --------------------------------------------------
    def recall(self, query: str, *, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve memories most similar to ``query``."""

        if not self.memories:
            return []
        qvec = self._embed(query)
        qvec = qvec.reshape(1, -1).astype(np.float32)
        k = min(top_k, len(self.memories))
        nearest = self.vector_store.find_nearest(qvec[0], k)
        mem_lookup = {m["id"]: m["text"] for m in self.memories}
        results: List[Dict[str, Any]] = []
        for pid, score in nearest:
            text = mem_lookup.get(pid, "")
            results.append({"id": pid, "text": text, "score": float(score)})
        return results

    # --------------------------------------------------
    def compress(
        self,
        text: str,
        budget: int,
        previous_compression_result: Optional[CompressedMemory] = None,
    ) -> CompressedMemory:
        """Naive compression via truncation."""

        truncated = text[:budget]
        trace = CompressionTrace(
            engine_name="base_truncate",  # Or perhaps self.id for consistency
            strategy_params={"budget": budget},
            input_summary={"original_length": len(text)},
            steps=[{"type": "truncate", "details": {"budget": budget}}],
            output_summary={"compressed_length": len(truncated)},
        )
        return CompressedMemory(
            text=truncated,
            trace=trace,
            engine_id=getattr(self, "id", self.__class__.__name__),
            engine_config=self.config,
        )

    # --------------------------------------------------
    def save(self, path: str) -> None:
        """Persist memories and embeddings to ``path``."""

        os.makedirs(path, exist_ok=True)
        manifest = {
            "engine_id": getattr(self, "id", self.__class__.__name__),
            "engine_class": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            "config": self.config,
        }
        with open(
            os.path.join(path, "engine_manifest.json"), "w", encoding="utf-8"
        ) as fh:
            json.dump(manifest, fh)
        with open(os.path.join(path, "entries.json"), "w", encoding="utf-8") as fh:
            json.dump(self.memories, fh)
        np.save(os.path.join(path, "embeddings.npy"), self.embeddings)

    # --------------------------------------------------
    def load(self, path: str) -> None:
        """Load engine state from ``path``."""

        with open(os.path.join(path, "entries.json"), "r", encoding="utf-8") as fh:
            self.memories = json.load(fh)
        embeddings = np.load(os.path.join(path, "embeddings.npy"))

        self.vector_store = InMemoryVectorStore(embedding_dim=embeddings.shape[1])

        # Rebuild memory_hashes and populate vector store
        self.memory_hashes = set()
        for mem_entry, vec in zip(self.memories, embeddings):
            chunk_hash = calculate_sha256(mem_entry["text"])
            self.memory_hashes.add(chunk_hash)
            proto = BeliefPrototype(prototype_id=mem_entry["id"], vector_row_index=0)
            self.vector_store.add_prototype(proto, vec.astype(np.float32))
            memory = RawMemory(
                memory_id=mem_entry["id"],
                raw_text_hash=chunk_hash,
                raw_text=mem_entry["text"],
                embedding=vec.astype(np.float32).tolist(),
            )
            self.vector_store.add_memory(memory)

    @property
    def embeddings(self) -> np.ndarray:
        """Return stored embedding vectors."""
        store_vectors = getattr(self.vector_store, "proto_vectors", None)
        if store_vectors is None:
            return np.zeros((0, get_embedding_dim()), dtype=np.float32)
        return store_vectors

    @property
    def chunker(self) -> Chunker:
        """Return the current :class:`Chunker`."""
        return self._chunker

    @chunker.setter
    def chunker(self, value: Chunker) -> None:
        if not isinstance(value, Chunker):
            raise TypeError("chunker must implement Chunker interface")
        self._chunker = value
        # If self.config exists and is a dict, update chunker_id
        if hasattr(self, "config") and isinstance(self.config, dict):
            self.config["chunker_id"] = getattr(value, "id", type(value).__name__)


__all__ = [
    "BaseCompressionEngine",
    "CompressedMemory",
    "CompressionTrace",
]
