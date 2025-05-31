from __future__ import annotations

import hashlib
import json
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, TypedDict, Callable

import logging
import numpy as np


from .chunker import Chunker, SentenceWindowChunker
from .embedding_pipeline import embed_text
from .json_npy_store import JsonNpyVectorStore, BeliefPrototype, RawMemory
from .memory_creation import (
    ExtractiveSummaryCreator,
    MemoryCreator,
)
from .prompt_budget import PromptBudget
from .token_utils import truncate_text, token_count
from .canonical import render_five_w_template
from .conflict_flagging import ConflictFlagger, ConflictLogger as FlagLogger
from .conflict import SimpleConflictLogger
from .active_memory_manager import ActiveMemoryManager, ConversationTurn


class VectorIndexCorrupt(RuntimeError):
    """Raised when prototype index and vectors are misaligned."""


class _LRUSet:
    """Simple fixed-size LRU cache for SHA hashes."""

    def __init__(self, size: int = 128) -> None:
        self.size = size
        # OrderedDict preserves insertion order and provides an efficient
        # LRU eviction mechanism via :meth:`popitem`.
        self._cache: "OrderedDict[str, None]" = OrderedDict()

    def add(self, item: str) -> bool:
        """Add ``item`` and return ``True`` if it was new."""
        if item in self._cache:
            # Touch to mark as recently used without counting as new
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
                    {
                        "belief_id": belief_id,
                        "memory_id": memory_id,
                        "weight": weight,
                    }
                )
                + "\n"
            )


class PrototypeHit(TypedDict):
    """Prototype search result."""

    id: str
    summary: str
    sim: float


class MemoryHit(TypedDict):
    """Individual memory search result."""

    id: str
    text: str
    sim: float


class QueryResult(TypedDict):
    """Return type for :meth:`Agent.query`."""

    prototypes: List[PrototypeHit]
    memories: List[MemoryHit]
    status: str


class Agent:
    """Core ingestion logic operating on a :class:`JsonNpyVectorStore`."""

    def __init__(
        self,
        store: JsonNpyVectorStore,
        *,
        chunker: Optional[Chunker] = None,
        similarity_threshold: float = 0.8,
        dedup_cache: int = 128,
        summary_creator: Optional[MemoryCreator] = None,
        update_summaries: bool = False,
        prompt_budget: PromptBudget | None = None,
    ) -> None:
        if not 0.5 <= similarity_threshold <= 0.95:
            raise ValueError("similarity_threshold must be between 0.5 and 0.95")
        self.store = store
        self.chunker = chunker or SentenceWindowChunker()
        self.similarity_threshold = similarity_threshold
        self.summary_creator = summary_creator or ExtractiveSummaryCreator(max_words=25)
        self.update_summaries = update_summaries
        self.metrics: Dict[str, int] = {
            "memories_ingested": 0,
            "prototypes_spawned": 0,
            "duplicates_skipped": 0,
        }
        self._dedup = _LRUSet(size=dedup_cache)
        if isinstance(store.path, (str, Path)):
            p = Path(store.path) / "evidence.jsonl"
            c = Path(store.path) / "conflicts.jsonl"
        else:
            p = Path("evidence.jsonl")
            c = Path("conflicts.jsonl")
        self._evidence = EvidenceWriter(p)
        if isinstance(store.path, (str, Path)):
            c = Path(store.path) / "conflicts.jsonl"
        else:
            c = Path("conflicts.jsonl")
        self._conflict_logger = SimpleConflictLogger(c)
        self._conflicts = ConflictFlagger(FlagLogger(c))
        self.prompt_budget = prompt_budget

    # ------------------------------------------------------------------
    def get_statistics(self) -> Dict[str, object]:
        """Return summary statistics about the current store."""

        from .utils import get_disk_usage

        path = Path(self.store.path)
        return {
            "prototypes": len(self.store.prototypes),
            "memories": len(self.store.memories),
            "tau": self.similarity_threshold,
            "updated": self.store.meta.get("updated_at"),
            "disk_usage": get_disk_usage(path),
        }

    # ------------------------------------------------------------------
    def get_prototypes_view(
        self, sort_by: str | None = None
    ) -> List[Dict[str, object]]:
        """Return list of prototypes for display purposes."""

        protos = list(self.store.prototypes)
        if sort_by == "strength":
            protos.sort(key=lambda p: p.strength, reverse=True)

        return [
            {
                "id": p.prototype_id,
                "strength": p.strength,
                "confidence": p.confidence,
                "summary": p.summary_text,
            }
            for p in protos
        ]

    # ------------------------------------------------------------------
    def _log_conflict(self, proto_id: str, mem_a: RawMemory, mem_b: RawMemory) -> None:
        self._conflict_logger.add(proto_id, mem_a, mem_b)

    # ------------------------------------------------------------------
    def _write_evidence(self, belief_id: str, mem_id: str, weight: float) -> None:
        self._evidence.add(belief_id, mem_id, weight)

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
        """Ingest ``text`` into the store and return per-chunk statuses."""

        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in self._dedup:
            self.metrics["duplicates_skipped"] += 1
            return [{"duplicate": True}]
        self._dedup.add(digest)

        chunks = self.chunker.chunk(text)
        if not chunks:
            return []
        canonical = [
            render_five_w_template(
                c, who=who, what=what, when=when, where=where, why=why
            )
            for c in chunks
        ]
        vecs = embed_text(canonical)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)

        results: List[Dict[str, object]] = []
        total = len(chunks)
        for idx, (chunk, vec) in enumerate(zip(chunks, vecs), 1):
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
            self._write_evidence(pid, mem_id, 1.0)
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
            return {
                "prototypes": [],
                "memories": [],
                "status": "no_match",
            }

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

        return {
            "prototypes": proto_results,
            "memories": mem_results,
            "status": "ok",
        }

    # ------------------------------------------------------------------
    def process_conversational_turn(
        self,
        input_message: str,
        manager: ActiveMemoryManager,
    ) -> tuple[str, dict]:
        """Generate a reply to ``input_message`` using ``manager`` for context."""

        from .local_llm import LocalChatModel

        llm = getattr(self, "_chat_model", None)
        if llm is None:
            llm = LocalChatModel()
            self._chat_model = llm

        vec = embed_text([input_message])
        if vec.ndim != 1:
            vec = vec.reshape(-1)

        history_candidates = manager.select_history_candidates_for_prompt(vec)
        candidate_tokens = token_count(
            llm.tokenizer, "\n".join(t.text for t in history_candidates)
        )
        logging.debug(
            "[prompt] history candidates=%d tokens=%d",
            len(history_candidates),
            candidate_tokens,
        )
        if hasattr(llm, "_context_length"):
            max_len = llm._context_length()
        else:  # fallback for test doubles
            cfg = getattr(getattr(llm, "model", None), "config", None)
            max_len = getattr(cfg, "n_positions", 1024)
        total_budget = max_len - llm.max_new_tokens

        budget_cfg = manager.prompt_budget or getattr(self, "prompt_budget", None)
        if budget_cfg is not None:
            budgets = budget_cfg.resolve(total_budget)
        else:
            budgets = {}

        b_query = budgets.get("query")
        b_recent = budgets.get("recent_history")
        b_older = budgets.get("older_history")
        b_ltm = budgets.get("ltm_snippets")

        def _fit(turns, limit):
            if limit is None:
                return list(turns)
            kept = []
            tokens_used = 0
            for t in turns:
                n = token_count(llm.tokenizer, t.text)
                if tokens_used + n <= limit:
                    kept.append(t)
                    tokens_used += n
                else:
                    break
            return kept

        num_recent = manager.config_prompt_num_forced_recent_turns
        if num_recent > 0:
            recent_slice = history_candidates[-num_recent:]
            older_slice = history_candidates[:-num_recent]
        else:
            recent_slice = []
            older_slice = list(history_candidates)

        older_final = _fit(older_slice, b_older)
        recent_final = _fit(recent_slice, b_recent)
        history_final = older_final + recent_final

        history_text = "\n".join(t.text for t in history_final)
        history_tokens_final = token_count(llm.tokenizer, history_text)
        logging.debug(
            "[prompt] history after fit turns=%d tokens=%d",
            len(history_final),
            history_tokens_final,
        )

        query_res = self.query(input_message, top_k_prototypes=2, top_k_memories=2)
        proto_summaries = "; ".join(
            p["summary"] for p in query_res.get("prototypes", [])
        )
        mem_texts = "; ".join(m["text"] for m in query_res.get("memories", []))
        ltm_initial = "\n".join(
            [
                f"Relevant concepts: {proto_summaries}" if proto_summaries else "",
                f"Context: {mem_texts}" if mem_texts else "",
            ]
        ).strip()
        logging.debug(
            "[prompt] LTM snippets tokens before=%d",
            token_count(llm.tokenizer, ltm_initial) if ltm_initial else 0,
        )

        user_text = input_message
        if b_query:
            logging.debug(
                "[prompt] query tokens before=%d limit=%s",
                token_count(llm.tokenizer, user_text),
                b_query,
            )
            user_text = truncate_text(llm.tokenizer, user_text, b_query)
            logging.debug(
                "[prompt] query tokens after=%d",
                token_count(llm.tokenizer, user_text),
            )

        ltm_parts = []
        if proto_summaries:
            ltm_parts.append(f"Relevant concepts: {proto_summaries}")
        if mem_texts:
            ltm_parts.append(f"Context: {mem_texts}")
        ltm_text = "\n".join(ltm_parts)
        if b_ltm:
            logging.debug(
                "[prompt] LTM tokens before=%d limit=%s",
                token_count(llm.tokenizer, ltm_text),
                b_ltm,
            )
            ltm_text = truncate_text(llm.tokenizer, ltm_text, b_ltm)
            logging.debug(
                "[prompt] LTM tokens after=%d",
                token_count(llm.tokenizer, ltm_text),
            )

        parts = []
        if history_text:
            parts.append(history_text)
        parts.append(f"User asked: {user_text}")
        if ltm_text:
            parts.append(ltm_text)
        parts.append("Answer:")
        prompt = "\n".join(parts)
        before_llm_tokens = token_count(llm.tokenizer, prompt)
        logging.debug("[prompt] before prepare tokens=%d", before_llm_tokens)
        prompt = llm.prepare_prompt(self, prompt)
        after_llm_tokens = token_count(llm.tokenizer, prompt)
        logging.debug("[prompt] after prepare tokens=%d", after_llm_tokens)
        reply = llm.reply(prompt)

        manager.add_turn(
            ConversationTurn(
                text=f"User: {input_message}\nAgent: {reply}",
                turn_embedding=vec.tolist() if hasattr(vec, "tolist") else None,
            )
        )

        return reply, {"query_result": query_res}

    # ------------------------------------------------------------------
    def receive_channel_message(
        self,
        source_id: str,
        message_text: str,
        manager: Optional[ActiveMemoryManager] = None,
    ) -> dict[str, object]:
        """Process ``message_text`` posted to the shared channel by ``source_id``.

        The default behaviour is intentionally simple:

        * If the message looks like a question (ends with ``?``) the agent
          performs a :meth:`query` and attempts to generate a short textual
          reply using :class:`~gist_memory.local_llm.LocalChatModel`.
        * Otherwise the message is ingested as a new memory with a
          ``source_document_id`` that references the sender.

        A dictionary summarising the chosen action is returned.  This provides
        lightweight observability for higher level session management code.
        """

        logging.info("[receive] from %s: %s", source_id, message_text[:40])

        summary: dict[str, object] = {"source": source_id}

        text = message_text.strip()
        if text.endswith("?"):
            # -------------------------------------------------- Option A: query
            if manager is not None:
                reply, info = self.process_conversational_turn(text, manager)
                summary["action"] = "query"
                summary["query_result"] = info.get("query_result", {})
                summary["reply"] = reply
                return summary

            result = self.query(text, top_k_prototypes=2, top_k_memories=2)
            summary["action"] = "query"
            summary["query_result"] = result
            reply: Optional[str] = None
            try:
                from .local_llm import LocalChatModel

                llm = getattr(self, "_chat_model", None)
                if llm is None:
                    llm = LocalChatModel()
                    self._chat_model = llm

                proto_summaries = "; ".join(
                    p["summary"] for p in result.get("prototypes", [])
                )
                mem_texts = "; ".join(m["text"] for m in result.get("memories", []))
                prompt_parts = [f"User asked: {text}"]
                if proto_summaries:
                    prompt_parts.append(f"Relevant concepts: {proto_summaries}")
                if mem_texts:
                    prompt_parts.append(f"Context: {mem_texts}")
                prompt_parts.append("Answer:")
                prompt = "\n".join(prompt_parts)
                prompt = llm.prepare_prompt(self, prompt)
                reply = llm.reply(prompt)
            except Exception as exc:  # pragma: no cover - optional dep
                logging.warning("chat reply failed: %s", exc)
                reply = None

            summary["reply"] = reply
            return summary

        # -------------------------------------------- Option B: ingest message
        results = self.add_memory(
            text,
            source_document_id=f"session_post_from:{source_id}",
        )
        summary.update(
            {
                "action": "ingest",
                "chunks_ingested": len(results),
                "reply": None,
            }
        )
        return summary
