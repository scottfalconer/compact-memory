from __future__ import annotations

"""Compression engine utilities and dataclasses."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence
import json
import os
import uuid

import numpy as np
import faiss

from ..chunker import SentenceWindowChunker
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
        chunker: SentenceWindowChunker | None = None,
        embedding_fn: Callable[[str | Sequence[str]], np.ndarray] = embed_text,
        preprocess_fn: Callable[[str], str] | None = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = {}
            if chunker:
                self.config['chunker_id'] = type(chunker).__name__
            # Note: embedding_fn and preprocess_fn are not easily serializable by default.
            # Subclasses would need to handle serialization/deserialization if they are configurable.

        self.chunker = chunker or SentenceWindowChunker()
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


def load_engine(path: str | os.PathLike) -> BaseCompressionEngine:
    """Load a compression engine from ``path`` using its manifest."""

    import importlib
    from pathlib import Path
    from ..engine_registry import get_compression_engine

    p = Path(path)
    with open(p / "engine_manifest.json", "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    engine_id = manifest.get("engine_id")
    engine_class = manifest.get("engine_class")
    engine_config = manifest.get("config", {})
    cls = None
    if engine_class:
        mod_name, cls_name = engine_class.rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        cls = getattr(mod, cls_name)
    elif engine_id:
        cls = get_compression_engine(engine_id)
    else:
        raise ValueError("Engine manifest missing engine_id or engine_class")

    # Instantiate the engine, passing the config if available
    if engine_config:
        engine = cls(**engine_config)
    else:
        engine = cls()

    engine.load(p)
    return engine


__all__ = [
    "BaseCompressionEngine",
    "CompressedMemory",
    "CompressionTrace",
    "load_engine",
]
