from __future__ import annotations

"""Prototype-based long-term memory as a CompressionStrategy."""

import hashlib
import json
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any, Tuple

import numpy as np

from .chunking import ChunkFn
from .memory_store import MemoryStore
from .models import BeliefPrototype, RawMemory
from .memory_creation import ExtractiveSummaryCreator, MemoryCreator
from .prototype.canonical import render_five_w_template
from .prototype.conflict_flagging import ConflictFlagger, ConflictLogger as FlagLogger
from .prototype.conflict import SimpleConflictLogger
from .compression.strategies_abc import (
    CompressedMemory,
    CompressionStrategy,
    CompressionTrace,
)  # Added CompressionTrace
from .token_utils import truncate_text


class _LRUSet:
    """Simple fixed-size LRU cache for SHA hashes."""

    def __init__(self, size: int = 128) -> None:
        self.size = size
        self._cache: "OrderedDict[str, None]" = OrderedDict()

    def add(self, item: str) -> bool:
        if item in self._cache:
            self._cache.move_to_end(item)
            return False
        self._cache[item] = None
        if len(self._cache) > self.size:
            self._cache.popitem(last=False)
        return True

    def __contains__(self, item: str) -> bool:
        return item in self._cache


class EvidenceWriter:
    """Append evidence rows to ``evidence.jsonl``."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, belief_id: str, memory_id: str, weight: float) -> None:
        with open(self.path, "a") as f:
            row = json.dumps(
                {"belief_id": belief_id, "memory_id": memory_id, "weight": weight}
            )
            f.write(row + "\n")


class PrototypeSystemStrategy(CompressionStrategy):
    """Prototype-based long-term memory management. Prototypes are updated over time using an EMA so their representations evolve with new evidence."""

    id = "prototype"

    def __init__(
        self,
        store: MemoryStore,
        *,
        chunk_fn: ChunkFn | None = None,
        similarity_threshold: float = 0.8,
        dedup_cache: int = 128,
        summary_creator: Optional[MemoryCreator] = None,
        update_summaries: bool = False,
    ) -> None:
        if not 0.5 <= similarity_threshold <= 0.95:
            raise ValueError("similarity_threshold must be between 0.5 and 0.95")
        self.store = store
        self.chunk_fn = chunk_fn
        self.similarity_threshold = float(similarity_threshold)
        self.summary_creator = summary_creator or ExtractiveSummaryCreator(max_words=25)
        self.update_summaries = update_summaries
        self.metrics: Dict[str, float] = {
            "memories_ingested": 0,
            "prototypes_spawned": 0,
            "duplicates_skipped": 0,
            "prototypes_updated": 0,
            "prototype_vector_change_magnitude": 0.0,
        }
        self._dedup = _LRUSet(size=dedup_cache)
        if isinstance(store.path, (str, Path)):
            p = Path(store.path) / "evidence.jsonl"
            c = Path(store.path) / "conflicts.jsonl"
        else:
            p = Path("evidence.jsonl")
            c = Path("conflicts.jsonl")
        self._evidence = EvidenceWriter(p)
        self._conflict_logger = SimpleConflictLogger(c)
        self._conflicts = ConflictFlagger(FlagLogger(c))

    # ------------------------------------------------------------------
    def add_memory(
        self,
        text: str,
        *,
        who: Optional[str] = None,
        what: Optional[str] = None,
        when: Optional[str] = None,
        where: Optional[str] = None,
        why: Optional[str] = None,
        progress_callback: Optional[
            Callable[[int, int, bool, str, Optional[float]], None]
        ] = None,
        save: bool = True,
        source_document_id: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        """Ingest ``text``. Related memories update existing prototypes to evolve their gist over time."""

        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in self._dedup:
            self.metrics["duplicates_skipped"] += 1
            return [{"duplicate": True}]
        self._dedup.add(digest)

        if self.chunk_fn is not None:
            chunks = self.chunk_fn(text)
        else:
            chunks = [text]
        if not chunks:
            return []
        canonical = [
            render_five_w_template(
                c, who=who, what=what, when=when, where=where, why=why
            )
            for c in chunks
        ]
        from . import agent as _agent

        vecs = _agent.embed_text(canonical)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)

        results: List[Dict[str, object]] = []
        total = len(chunks)
        for idx, (chunk, vec) in enumerate(zip(chunks, vecs), 1):
            mem_id = str(uuid.uuid4())
            nearest = self.store.find_nearest(vec, k=1)
            if not nearest and len(self.store.prototypes) > 0:
                raise RuntimeError("prototype index inconsistent")
            spawned = False
            sim: Optional[float] = None
            if nearest:
                pid, sim = nearest[0]
            if nearest and sim is not None and sim >= self.similarity_threshold:
                change = self.store.update_prototype(pid, vec, mem_id)
                self.metrics["prototypes_updated"] += 1
                n = self.metrics["prototypes_updated"]
                prev = self.metrics.get("prototype_vector_change_magnitude", 0.0)
                self.metrics["prototype_vector_change_magnitude"] = (
                    prev * (n - 1) + change
                ) / n
            else:
                summary = self.summary_creator.create(chunk)
                proto = BeliefPrototype(
                    prototype_id=str(uuid.uuid4()),
                    vector_row_index=0,
                    summary_text=summary,
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
                source_document_id=source_document_id,
                raw_text=chunk,
                embedding=list(map(float, vec)),
            )
            self.store.add_memory(raw_mem)
            self._evidence.add(pid, mem_id, 1.0)
            self._flag_conflicts(pid, raw_mem, vec)
            self.metrics["memories_ingested"] += 1
            if self.update_summaries:
                texts = [
                    m.raw_text
                    for m in self.store.memories
                    if m.assigned_prototype_id == pid
                ][:5]
                words = " ".join(texts).split()
                summary = " ".join(words[:25])
                for p in self.store.prototypes:
                    if p.prototype_id == pid:
                        p.summary_text = summary
                        break
            results.append({"prototype_id": pid, "spawned": spawned, "sim": sim})
            if progress_callback:
                progress_callback(idx, total, spawned, pid, sim)

        if save:
            self.store.save()
        return results

    # ------------------------------------------------------------------
    def _flag_conflicts(
        self,
        prototype_id: str,
        new_mem: RawMemory,
        new_vec: np.ndarray,
    ) -> None:
        for mem in self.store.memories:
            if mem.assigned_prototype_id != prototype_id:
                continue
            if mem.memory_id == new_mem.memory_id:
                continue
            if mem.embedding is None:
                continue
            vec_b = np.array(mem.embedding, dtype=np.float32)
            self._conflicts.check_pair(prototype_id, new_mem, new_vec, mem, vec_b)

    # ------------------------------------------------------------------
    def query(
        self,
        text: str,
        *,
        top_k_prototypes: int = 1,
        top_k_memories: int = 3,
        include_hypotheses: bool = False,
    ) -> Dict[str, object]:
        """Return nearest prototypes and memories for ``text``."""

        from . import agent as _agent

        vec = _agent.embed_text(text)
        if vec.ndim != 1:
            vec = vec.reshape(-1)
        nearest = self.store.find_nearest(vec, k=top_k_prototypes)
        if not nearest:
            return {"prototypes": [], "memories": [], "status": "no_match"}

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

        return {"prototypes": proto_results, "memories": mem_results, "status": "ok"}

    # ------------------------------------------------------------------
    def compress(
        self,
        text: str,  # Changed
        llm_token_budget: int,
        chunk_fn: Optional[ChunkFn] = None,  # Added
        *,
        tokenizer=None,
        **kwargs: Any,  # Added for consistency
    ) -> Tuple[CompressedMemory, CompressionTrace]:  # Changed return type
        """Compress by retrieving relevant memories and truncating."""

        if chunk_fn:  # This is the chunk_fn passed to compress
            chunks = chunk_fn(text)
            query_text = " ".join(chunks)
            num_input_chunks = len(chunks)
        else:
            chunks = [text]  # Conceptually, for trace
            query_text = text
            num_input_chunks = 1

        original_input_len = len(query_text)

        result = self.query(query_text, top_k_prototypes=1, top_k_memories=3)

        retrieved_prototypes = result.get("prototypes", [])
        retrieved_memories = result.get("memories", [])

        proto_summaries = "; ".join(p["summary"] for p in retrieved_prototypes)
        mem_texts = "; ".join(m["text"] for m in retrieved_memories)
        combined = " ".join(filter(None, [proto_summaries, mem_texts]))

        if tokenizer is not None:
            compressed = truncate_text(tokenizer, combined, llm_token_budget)
        else:
            # Simple character truncation if no tokenizer
            compressed = combined[:llm_token_budget]

        compressed_memory = CompressedMemory(
            text=compressed,
            metadata={
                "status": result.get("status", "unknown"),
                "retrieved_prototypes_count": len(retrieved_prototypes),
                "retrieved_memories_count": len(retrieved_memories),
            },
        )

        trace = CompressionTrace(
            strategy_name=self.id,
            strategy_params={
                "llm_token_budget": llm_token_budget,
                "chunked_input_to_compress": chunk_fn is not None,
                "top_k_prototypes_query": 1,  # Hardcoded in query call
                "top_k_memories_query": 3,  # Hardcoded in query call
            },
            input_summary={
                "input_query_char_len": original_input_len,
                "input_query_num_chunks": num_input_chunks,
            },
            steps=[
                {
                    "type": "query_prototype_store",
                    "query_text_preview": query_text[:100],
                    "retrieved_prototypes": len(retrieved_prototypes),
                    "retrieved_memories": len(retrieved_memories),
                },
                {"type": "concatenation", "combined_char_len": len(combined)},
                {"type": "truncation", "final_char_len": len(compressed)},
            ],
            output_summary={"compressed_char_len": len(compressed)},
            final_compressed_object_preview=compressed[:50],
        )

        return compressed_memory, trace
