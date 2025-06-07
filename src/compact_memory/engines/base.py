from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence
import json
import os
import uuid

import numpy as np
import faiss

from ..chunker import Chunker, SentenceWindowChunker
from ..embedding_pipeline import embed_text, get_embedding_dim


@dataclass
class CompressedMemory:
    """Container for compressed text and metadata."""

    text: str
    metadata: Optional[Dict[str, Any]] = None


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
        config: Optional[Dict[str, Any]] = None,
        **kwargs, # Allow other config params to be passed
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = kwargs # Store kwargs if no explicit config dict is given

        # Prioritize chunker_id from config if it exists.
        # Otherwise, if a chunker instance is passed, derive its ID.
        # Otherwise, set a default chunker_id.
        if 'chunker_id' not in self.config:
            if chunker:
                self.config['chunker_id'] = type(chunker).__name__
            else:
                self.config['chunker_id'] = type(SentenceWindowChunker()).__name__
        # elif chunker and type(chunker).__name__ != self.config['chunker_id']:
            # This case could be a warning or error if a chunker instance is passed
            # that doesn't match a pre-existing chunker_id in the config.
            # For now, we assume config['chunker_id'] (if present) is the source of truth.

            # Note: embedding_fn and preprocess_fn are not easily serializable by default.
            # Subclasses would need to handle serialization/deserialization if they are configurable.

        self._chunker = chunker or SentenceWindowChunker()
        self.embedding_fn = embedding_fn
        self.preprocess_fn = preprocess_fn
        self.memories: List[Dict[str, Any]] = []
        dim = get_embedding_dim()
        self.embeddings = np.zeros((0, dim), dtype=np.float32)
        self.index: faiss.IndexFlatIP | None = None

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
        if self.index is None and self.embeddings.size:
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.index.add(self.embeddings.astype(np.float32))

    # --------------------------------------------------
    def ingest(self, text: str) -> List[str]:
        """Chunk and store ``text`` in the engine."""

        raw_chunks = self.chunker.chunk(text)
        if not raw_chunks:
            return []

        processed_chunks = [self._compress_chunk(chunk) for chunk in raw_chunks]

        vecs = self.embedding_fn(processed_chunks, preprocess_fn=self.preprocess_fn)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        ids: List[str] = []
        for processed_chunk_text, vec in zip(processed_chunks, vecs):
            mid = uuid.uuid4().hex
            self.memories.append({"id": mid, "text": processed_chunk_text})
            self.embeddings = np.vstack([self.embeddings, vec.astype(np.float32)])
            ids.append(mid)
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(vecs.astype(np.float32))
        return ids

    # --------------------------------------------------
    def recall(self, query: str, *, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve memories most similar to ``query``."""

        if not self.memories:
            return []
        self._ensure_index()
        qvec = self.embedding_fn(query, preprocess_fn=self.preprocess_fn)
        qvec = qvec.reshape(1, -1).astype(np.float32)
        k = min(top_k, len(self.memories))
        dists, idxs = self.index.search(qvec, k)
        results: List[Dict[str, Any]] = []
        for idx, dist in zip(idxs[0], dists[0]):
            if idx < 0:
                continue
            mem = self.memories[int(idx)]
            results.append({"id": mem["id"], "text": mem["text"], "score": float(dist)})
        return results

    # --------------------------------------------------
    def compress(
        self, text: str, budget: int
    ) -> tuple[CompressedMemory, CompressionTrace]:
        """Naive compression via truncation."""

        truncated = text[:budget]
        trace = CompressionTrace(
            engine_name="base_truncate",
            strategy_params={"budget": budget},
            input_summary={"original_length": len(text)},
            steps=[{"type": "truncate", "details": {"budget": budget}}],
            output_summary={"compressed_length": len(truncated)},
        )
        return CompressedMemory(text=truncated), trace

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
        self.embeddings = np.load(os.path.join(path, "embeddings.npy"))
        self.index = None
        self._ensure_index()

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
        if hasattr(self, 'config') and isinstance(self.config, dict):
            self.config['chunker_id'] = getattr(value, "id", type(value).__name__)

__all__ = [
    "BaseCompressionEngine",
    "CompressedMemory",
    "CompressionTrace",
]
