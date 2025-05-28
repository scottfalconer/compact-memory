from __future__ import annotations

import hashlib
import json
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import logging
import numpy as np


from .chunker import Chunker, SentenceWindowChunker
from .embedding_pipeline import embed_text
from .json_npy_store import JsonNpyVectorStore, BeliefPrototype, RawMemory


class VectorIndexCorrupt(RuntimeError):
    """Raised when prototype index and vectors are misaligned."""


class _LRUSet:
    """Simple fixed-size LRU cache for SHA hashes."""

    def __init__(self, size: int = 128) -> None:
        self.size = size
        self._queue: deque[str] = deque(maxlen=size)
        self._set: set[str] = set()

    def add(self, item: str) -> bool:
        """Add ``item`` and return ``True`` if it was new."""
        if item in self._set:
            return False
        if len(self._queue) == self.size:
            old = self._queue.popleft()
            self._set.remove(old)
        self._queue.append(item)
        self._set.add(item)
        return True

    def __contains__(self, item: str) -> bool:
        return item in self._set


class EvidenceWriter:
    """Append evidence rows to ``evidence.jsonl``."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, belief_id: str, memory_id: str, weight: float) -> None:
        with open(self.path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "belief_id": belief_id,
                        "memory_id": memory_id,
                        "weight": weight,
                    }
                )
                + "\n"
            )


@dataclass
class QueryResult:
    """Return type for :meth:`Agent.query`."""

    prototypes: List[Dict[str, object]]
    memories: List[Dict[str, object]]
    status: str = "ok"


class Agent:
    """Core ingestion logic operating on a :class:`JsonNpyVectorStore`."""

    def __init__(
        self,
        store: JsonNpyVectorStore,
        *,
        chunker: Optional[Chunker] = None,
        similarity_threshold: float = 0.8,
        dedup_cache: int = 128,
    ) -> None:
        if not 0.5 <= similarity_threshold <= 0.95:
            raise ValueError("similarity_threshold must be between 0.5 and 0.95")
        self.store = store
        self.chunker = chunker or SentenceWindowChunker()
        self.similarity_threshold = similarity_threshold
        self.metrics: Dict[str, int] = {
            "memories_ingested": 0,
            "prototypes_spawned": 0,
            "duplicates_skipped": 0,
        }
        self._dedup = _LRUSet(size=dedup_cache)
        if isinstance(store.path, (str, Path)):
            p = Path(store.path) / "evidence.jsonl"
        else:
            p = Path("evidence.jsonl")
        self._evidence = EvidenceWriter(p)

    # ------------------------------------------------------------------
    def _write_evidence(self, belief_id: str, mem_id: str, weight: float) -> None:
        self._evidence.add(belief_id, mem_id, weight)

    # ------------------------------------------------------------------
    def add_memory(self, text: str) -> List[Dict[str, object]]:
        """Ingest ``text`` into the store and return per-chunk statuses."""

        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in self._dedup:
            self.metrics["duplicates_skipped"] += 1
            return [{"duplicate": True}]
        self._dedup.add(digest)

        chunks = self.chunker.chunk(text)
        if not chunks:
            return []
        vecs = embed_text(chunks)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)

        results: List[Dict[str, object]] = []
        for chunk, vec in zip(chunks, vecs):
            mem_id = str(uuid.uuid4())
            nearest = self.store.find_nearest(vec, k=1)
            if not nearest and len(self.store.prototypes) > 0:
                raise VectorIndexCorrupt("prototype index inconsistent")
            spawned = False
            sim: Optional[float] = None
            if nearest:
                pid, sim = nearest[0]
            if nearest and sim is not None and sim >= self.similarity_threshold:
                self.store.update_prototype(pid, vec, mem_id)
            else:
                # TODO density guard hook
                proto = BeliefPrototype(
                    prototype_id=str(uuid.uuid4()),
                    vector_row_index=0,
                    summary_text="",
                    strength=1.0,
                    confidence=1.0,
                    constituent_memory_ids=[mem_id],
                )
                self.store.add_prototype(proto, vec)
                pid = proto.prototype_id
                spawned = True
                self.metrics["prototypes_spawned"] += 1

            raw_mem = RawMemory(
                memory_id=mem_id,
                raw_text_hash=hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
                assigned_prototype_id=pid,
                source_document_id=None,
                raw_text=chunk,
                embedding=list(map(float, vec)),
            )
            self.store.add_memory(raw_mem)
            self._write_evidence(pid, mem_id, 1.0)
            self.metrics["memories_ingested"] += 1
            results.append({"prototype_id": pid, "spawned": spawned, "sim": sim})

        self.store.save()
        return results

    # ------------------------------------------------------------------
    def query(
        self,
        text: str,
        *,
        top_k_prototypes: int = 1,
        top_k_memories: int = 3,
        include_hypotheses: bool = False,
    ) -> QueryResult:
        """Return nearest prototypes and memories for ``text``."""

        vec = embed_text(text)
        if vec.ndim != 1:
            vec = vec.reshape(-1)
        nearest = self.store.find_nearest(vec, k=top_k_prototypes)
        if not nearest:
            return QueryResult(prototypes=[], memories=[], status="no_match")

        logging.info(
            "[query] '%s' → %d protos, top sim %.2f",
            text[:40],
            len(nearest),
            nearest[0][1],
        )

        proto_map = {p.prototype_id: p for p in self.store.prototypes}
        proto_results: List[Dict[str, object]] = []
        memory_candidates: List[tuple[float, RawMemory]] = []

        for pid, sim in nearest:
            proto = proto_map.get(pid)
            if not proto:
                continue
            proto_results.append({"id": pid, "summary": proto.summary_text, "sim": sim})
            for mid in proto.constituent_memory_ids:
                mem = next((m for m in self.store.memories if m.memory_id == mid), None)
                if mem is None:
                    continue
                if mem.embedding is not None:
                    mem_vec = np.array(mem.embedding, dtype=np.float32)
                    mem_sim = float(np.dot(vec, mem_vec))
                else:
                    mem_sim = float(sim)
                memory_candidates.append((mem_sim, mem))

        memory_candidates.sort(key=lambda x: -x[0])
        mem_results = [
            {"id": m.memory_id, "text": m.raw_text, "sim": s}
            for s, m in memory_candidates[:top_k_memories]
        ]

        return QueryResult(prototypes=proto_results, memories=mem_results)
