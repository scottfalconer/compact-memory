from __future__ import annotations

"""Prototype-based long-term memory as a CompressionStrategy."""

import hashlib
import json
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union # Added Sequence, Union

import numpy as np

from .chunker import Chunker, SentenceWindowChunker
from .vector_store import BaseVectorStore # Changed from InMemoryVectorStore
from .models import BeliefPrototype, RawMemory # Changed import location
from .memory_creation import ExtractiveSummaryCreator, MemoryCreator
from .prototype.canonical import render_five_w_template
from .prototype.conflict_flagging import ConflictFlagger, ConflictLogger as FlagLogger # Assuming these are still valid
from .prototype.conflict import SimpleConflictLogger # Assuming these are still valid
from .compression.strategies_abc import CompressedMemory, CompressionStrategy
from .token_utils import truncate_text
from .embedding_pipeline import embed_text, EmbeddingFunction # Import new embed_text and type

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
            f.write(
                json.dumps(
                    {"belief_id": belief_id, "memory_id": memory_id, "weight": weight}
                )
                + "\n"
            )

class PrototypeSystemStrategy(CompressionStrategy):
    id = "prototype"

    def __init__(
        self,
        store: BaseVectorStore, # Changed from InMemoryVectorStore
        embedding_dim: int,    # Added
        embedding_fn: Optional[EmbeddingFunction] = None, # Added
        *,
        chunker: Optional[Chunker] = None,
        similarity_threshold: float = 0.8,
        dedup_cache: int = 128,
        summary_creator: Optional[MemoryCreator] = None,
        update_summaries: bool = False,
        # TODO: Configure paths for evidence/conflict logs if needed
        # evidence_log_path: Optional[Path] = None,
        # conflict_log_path: Optional[Path] = None,
    ) -> None:
        if not 0.5 <= similarity_threshold <= 0.95:
            raise ValueError("similarity_threshold must be between 0.5 and 0.95")
        self.store = store
        self.embedding_dim = embedding_dim
        self.embedding_fn = embedding_fn # Store the embedding function
        self.chunker = chunker or SentenceWindowChunker()
        self.similarity_threshold = float(similarity_threshold)
        self.summary_creator = summary_creator or ExtractiveSummaryCreator(max_words=25)
        self.update_summaries = update_summaries

        self.prototypes: Dict[str, BeliefPrototype] = {}
        self.memories: Dict[str, RawMemory] = {}

        self.metrics: Dict[str, float] = {
            "memories_ingested": 0,
            "prototypes_spawned": 0,
            "duplicates_skipped": 0,
            "prototypes_updated": 0,
            "prototype_vector_change_magnitude": 0.0,
        }
        self._dedup = _LRUSet(size=dedup_cache)

        # Initialize evidence and conflict logging (paths need proper configuration)
        # For now, disabling them or using default local paths if absolutely needed.
        # These should ideally be passed in or configured via a settings object.
        self._evidence: Optional[EvidenceWriter] = None
        self._conflict_logger: Optional[SimpleConflictLogger] = None
        self._conflicts: Optional[ConflictFlagger] = None
        # Example: if evidence_log_path: self._evidence = EvidenceWriter(evidence_log_path)
        # if conflict_log_path:
        #     self._conflict_logger = SimpleConflictLogger(conflict_log_path)
        #     self._conflicts = ConflictFlagger(FlagLogger(conflict_log_path))


    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalizes a single vector."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def add_memory(
        self,
        text: str,
        *,
        who: Optional[str] = None,
        what: Optional[str] = None,
        when: Optional[str] = None,
        where: Optional[str] = None,
        why: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, bool, str, Optional[float]], None]] = None,
        # save parameter removed, Agent handles persistence
        source_document_id: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in self._dedup:
            self.metrics["duplicates_skipped"] += 1
            return [{"duplicate": True}]
        self._dedup.add(digest)

        chunks = self.chunker.chunk(text)
        if not chunks:
            return []

        canonical_chunks = [
            render_five_w_template(c, who=who, what=what, when=when, where=where, why=why)
            for c in chunks
        ]

        # Use the new embed_text function, passing self.embedding_fn
        vecs = embed_text(canonical_chunks, embedding_fn=self.embedding_fn)
        if vecs.ndim == 1 and len(canonical_chunks) == 1: # Single chunk resulted in 1D array
            vecs = vecs.reshape(1, -1)
        elif vecs.ndim == 1 and len(canonical_chunks) > 1: # Should not happen if embed_text is correct
             raise ValueError("Embedding multiple chunks should result in a 2D array.")


        results: List[Dict[str, object]] = []
        total = len(chunks)
        for idx, (chunk, vec) in enumerate(zip(chunks, vecs), 1): # vec is new_mem_vec
            mem_id = str(uuid.uuid4())
            normalized_mem_vec = self._normalize_vector(vec.astype(np.float32))

            nearest_query_results = self.store.query_vector(normalized_mem_vec, top_k=1)

            spawned = False
            sim: Optional[float] = None
            pid: Optional[str] = None

            if nearest_query_results:
                pid_candidate, sim_candidate = nearest_query_results[0]
                if sim_candidate >= self.similarity_threshold:
                    pid = pid_candidate
                    sim = sim_candidate

                    prototype_obj = self.prototypes.get(pid)
                    if prototype_obj:
                        alpha = 1.0 / (prototype_obj.strength + 1.0)
                        current_proto_vec = self.store.get_vector(pid) # Already normalized from store
                        updated_vec_unnormalized = (1 - alpha) * current_proto_vec + alpha * normalized_mem_vec
                        updated_vec = self._normalize_vector(updated_vec_unnormalized)

                        self.store.add_vector(pid, updated_vec, metadata={"summary": prototype_obj.summary_text})
                        change = float(np.linalg.norm(updated_vec - current_proto_vec))

                        prototype_obj.strength += 1
                        prototype_obj.constituent_memory_ids.append(mem_id)
                        prototype_obj.last_updated_ts = BeliefPrototype().last_updated_ts

                        self.metrics["prototypes_updated"] += 1
                        n_updates = self.metrics["prototypes_updated"]
                        prev_change_mag = self.metrics.get("prototype_vector_change_magnitude", 0.0)
                        self.metrics["prototype_vector_change_magnitude"] = (
                            (prev_change_mag * (n_updates - 1) + change) / n_updates
                        )
                    else:
                        pid = None

            if pid is None:
                summary = self.summary_creator.create(chunk)
                new_proto_id = str(uuid.uuid4())
                proto = BeliefPrototype(
                    prototype_id=new_proto_id,
                    summary_text=summary,
                    strength=1.0,
                    confidence=1.0,
                    constituent_memory_ids=[mem_id],
                )
                self.prototypes[new_proto_id] = proto
                self.store.add_vector(new_proto_id, normalized_mem_vec, metadata={"summary": proto.summary_text})
                pid = new_proto_id
                spawned = True
                self.metrics["prototypes_spawned"] += 1

            raw_mem = RawMemory(
                memory_id=mem_id,
                raw_text_hash=hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
                assigned_prototype_id=pid,
                source_document_id=source_document_id,
                raw_text=chunk,
                embedding=normalized_mem_vec.tolist(), # Store normalized embedding
            )
            self.memories[mem_id] = raw_mem

            if self._evidence and pid:
                self._evidence.add(pid, mem_id, 1.0) # type: ignore
            if self._conflicts and pid:
                self._flag_conflicts_base_store(pid, raw_mem, normalized_mem_vec)

            self.metrics["memories_ingested"] += 1

            if self.update_summaries and pid:
                prototype_obj = self.prototypes.get(pid)
                if prototype_obj:
                    constituent_texts = [
                        self.memories[m_id].raw_text
                        for m_id in prototype_obj.constituent_memory_ids
                        if m_id in self.memories
                    ][:5]
                    new_summary_text = self.summary_creator.create(" ".join(constituent_texts))
                    if prototype_obj.summary_text != new_summary_text:
                        prototype_obj.summary_text = new_summary_text
                        current_vector = self.store.get_vector(pid) # Vector is already normalized
                        self.store.add_vector(pid, current_vector, metadata={"summary": prototype_obj.summary_text})

            results.append({"prototype_id": pid, "spawned": spawned, "sim": sim})
            if progress_callback:
                progress_callback(idx, total, spawned, pid or "", sim)
        return results

    def _flag_conflicts_base_store(
        self,
        prototype_id: str,
        new_mem: RawMemory,
        new_vec: np.ndarray, # Should be normalized
    ) -> None:
        if not self._conflicts: return
        for mem_id_key, mem_obj in self.memories.items():
            if mem_obj.assigned_prototype_id != prototype_id:
                continue
            if mem_obj.memory_id == new_mem.memory_id:
                continue
            if mem_obj.embedding is None:
                continue
            vec_b = np.array(mem_obj.embedding, dtype=np.float32) # Already normalized if stored correctly
            self._conflicts.check_pair(prototype_id, new_mem, new_vec, mem_obj, vec_b) # type: ignore

    def query(
        self,
        text: str,
        *,
        top_k_prototypes: int = 1,
        top_k_memories: int = 3,
        include_hypotheses: bool = False, # This param is not used currently
    ) -> Dict[str, object]:
        query_vec_unnormalized = embed_text(text, embedding_fn=self.embedding_fn)
        if query_vec_unnormalized.ndim != 1: # Should be 1D for single query text
            query_vec_unnormalized = query_vec_unnormalized.reshape(-1)

        query_vec = self._normalize_vector(query_vec_unnormalized.astype(np.float32))

        nearest_proto_query_results = self.store.query_vector(query_vec, top_k=top_k_prototypes)
        if not nearest_proto_query_results:
            return {"prototypes": [], "memories": [], "status": "no_match"}

        proto_results: List[Dict[str, object]] = []
        memory_candidates: List[tuple[float, RawMemory]] = []

        for pid, sim_to_query in nearest_proto_query_results: # sim_to_query is with normalized prototype vec
            proto_obj = self.prototypes.get(pid)
            if not proto_obj:
                continue
            proto_results.append({"id": pid, "summary": proto_obj.summary_text, "sim": sim_to_query})

            for mem_id in proto_obj.constituent_memory_ids:
                mem_obj = self.memories.get(mem_id)
                if mem_obj is None or mem_obj.embedding is None:
                    continue

                mem_embedding_np = np.array(mem_obj.embedding, dtype=np.float32) # Assumed normalized
                mem_sim_to_query = float(np.dot(query_vec, mem_embedding_np))
                memory_candidates.append((mem_sim_to_query, mem_obj))

        memory_candidates.sort(key=lambda x: -x[0]) # Sort by similarity desc
        mem_results = [
            {"id": m.memory_id, "text": m.raw_text, "sim": s}
            for s, m in memory_candidates[:top_k_memories]
        ]
        return {"prototypes": proto_results, "memories": mem_results, "status": "ok"}

    def compress(
        self,
        text_or_chunks: Union[str, List[str]],
        llm_token_budget: int,
        *,
        tokenizer=None, # TODO: Define tokenizer type
    ) -> CompressedMemory:
        if isinstance(text_or_chunks, list):
            query_text = " ".join(text_or_chunks)
        else:
            query_text = text_or_chunks

        result = self.query(query_text, top_k_prototypes=1, top_k_memories=3)
        proto_summaries = "; ".join(p["summary"] for p in result.get("prototypes", [])) # type: ignore
        mem_texts = "; ".join(m["text"] for m in result.get("memories", [])) # type: ignore
        combined = " ".join(filter(None, [proto_summaries, mem_texts]))

        if tokenizer is not None:
            compressed_text = truncate_text(tokenizer, combined, llm_token_budget)
        else:
            # Fallback if no tokenizer, truncate by characters (approximate)
            # This might not be ideal for token budgets.
            char_budget = llm_token_budget * 4 # Rough estimate
            compressed_text = combined[:char_budget]

        return CompressedMemory(text=compressed_text, metadata={"status": result["status"]}) # type: ignore

    def save_state(self, path: Path) -> None:
        """Saves the state of prototypes and memories managed by this strategy."""
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "pss_prototypes.json", "w") as f:
            # Using model_dump for Pydantic v2
            json.dump({pid: p.model_dump() for pid, p in self.prototypes.items()}, f, default=str) # default=str for datetime

        with open(path / "pss_memories.json", "w") as f:
            json.dump({mid: m.model_dump() for mid, m in self.memories.items()}, f, default=str)

    def load_state(self, path: Path) -> None:
        """Loads the state of prototypes and memories managed by this strategy."""
        prototypes_path = path / "pss_prototypes.json"
        if prototypes_path.exists():
            with open(prototypes_path, "r") as f:
                prototypes_data = json.load(f)
                self.prototypes = {
                    # Using model_validate for Pydantic v2
                    pid: BeliefPrototype.model_validate(p_dict)
                    for pid, p_dict in prototypes_data.items()
                }

        memories_path = path / "pss_memories.json"
        if memories_path.exists():
            with open(memories_path, "r") as f:
                memories_data = json.load(f)
                self.memories = {
                    mid: RawMemory.model_validate(m_dict)
                    for mid, m_dict in memories_data.items()
                }
        # Note: vector_row_index in BeliefPrototype is not used by this version of PSS.
        # If loading data from an older version, it will be ignored.
        # Make sure the vector store is also loaded by the Agent.
