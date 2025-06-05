from __future__ import annotations

from typing import Dict, List, Optional, TypedDict, Callable

import hashlib
import uuid
from collections import OrderedDict
import numpy as np

import logging
from pathlib import Path


from .chunker import Chunker, SentenceWindowChunker
from .embedding_pipeline import embed_text
from .vector_store import VectorStore
from .prototype_system_utils import render_five_w_template
from .memory_creation import (
    ExtractiveSummaryCreator,
    MemoryCreator,
)
from .models import BeliefPrototype, RawMemory
from .prompt_budget import PromptBudget
from .token_utils import truncate_text, token_count
from compact_memory.contrib import ActiveMemoryManager
from compact_memory.contrib import ConversationTurn
from compact_memory.engines import (
    BaseCompressionEngine,
    CompressedMemory,
    CompressionTrace,
)
from compact_memory.engines.no_compression_engine import NoCompressionEngine


class VectorIndexCorrupt(RuntimeError):
    """Raised when prototype index and vectors are misaligned."""


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


class QueryResult(dict):
    """Result object returned by :meth:`PrototypeEngine.query` with HTML representation."""

    prototypes: List[PrototypeHit]
    memories: List[MemoryHit]
    status: str

    def __init__(
        self, prototypes: List[PrototypeHit], memories: List[MemoryHit], status: str
    ) -> None:
        super().__init__(prototypes=prototypes, memories=memories, status=status)

    # --------------------------------------------------------------
    def _repr_html_(self) -> str:
        proto_rows = "".join(
            f"<tr><td>{p['id']}</td><td>{p['summary']}</td><td>{p['sim']:.2f}</td></tr>"
            for p in self.get("prototypes", [])
        )
        mem_rows = "".join(
            f"<tr><td>{m['id']}</td><td>{m['text']}</td><td>{m['sim']:.2f}</td></tr>"
            for m in self.get("memories", [])
        )
        html = """<h4>Prototypes</h4><table><tr><th>ID</th><th>Summary</th><th>Sim</th></tr>{p_rows}</table>""".format(
            p_rows=proto_rows
        )
        html += """<h4>Memories</h4><table><tr><th>ID</th><th>Text</th><th>Sim</th></tr>{m_rows}</table>""".format(
            m_rows=mem_rows
        )
        return html


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


class PrototypeEngine(BaseCompressionEngine):
    """
    Core component for managing and interacting with a memory store.

    The PrototypeEngine class encapsulates the logic for ingesting text into a
    `VectorStore`, managing memory prototypes, querying the memory,
    and processing conversational turns with optional compression.

    This engine directly manages memory prototypes, handling consolidation and
    retrieval. Developers typically interact with it by providing a pre-configured
    store and then using its methods like ``add_memory``, ``query`` and
    ``receive_channel_message``.

    Attributes:
        store (VectorStore): The underlying vector store for memories and prototypes.
        metrics (Dict[str, Any]): A dictionary of metrics collected during operations.
        prompt_budget (Optional[PromptBudget]): Configuration for managing prompt sizes
                                             when interacting with LLMs.
    """

    def __init__(
        self,
        store: Optional[VectorStore] = None, # Made optional for loading from config
        *,
        chunker: Optional[Chunker] = None,
        similarity_threshold: float = 0.8, # Default if not in config
        dedup_cache: int = 128, # Default if not in config
        summary_creator: Optional[MemoryCreator] = None, # Default if not in config
        update_summaries: bool = False, # Default if not in config
        prompt_budget: PromptBudget | None = None,
        preprocess_fn: Callable[[str], str] | None = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """Initialize the engine."""

        # Config for superclass. If config is passed, use it, else use kwargs.
        # BaseCompressionEngine.__init__ will set self.config.
        # If a chunker is passed directly, it takes precedence.
        # self.config['chunker_id'] will be set by super() if chunker is provided and no config is.
        actual_chunker = chunker or SentenceWindowChunker()
        super().__init__(
            chunker=actual_chunker,
            preprocess_fn=preprocess_fn,
            config=config if config is not None else kwargs
        )

        # Initialize store
        if store is not None:
            self.store = store
        else:
            # If store is not provided (e.g., when loading), create a default one.
            # load() will populate it.
            from compact_memory.embedding_pipeline import get_embedding_dim
            from compact_memory.vector_store import InMemoryVectorStore
            # Try to get embedding_dim from config if saved, else default.
            embedding_dim = self.config.get('store_embedding_dim', get_embedding_dim())
            self.store = InMemoryVectorStore(embedding_dim=embedding_dim)
            if 'store_tau' in self.config: # Restore tau if it was saved for the store
                 self.store.meta['tau'] = self.config['store_tau']


        # Initialize attributes from self.config (set by super) or defaults
        # similarity_threshold has a setter that also updates store.meta['tau']
        self.similarity_threshold = float(self.config.get('similarity_threshold', similarity_threshold))
        if not 0.5 <= self.similarity_threshold <= 0.95:
            raise ValueError("similarity_threshold must be between 0.5 and 0.95")

        # summary_creator: For now, use direct param or default. Configurable summary_creator by ID is future work.
        # If summary_creator_id is in config, we might use a factory here.
        self.summary_creator = summary_creator or ExtractiveSummaryCreator(max_words=25)
        if 'summary_creator_id' in self.config and summary_creator is None:
            # Basic example: if we had a registry. For now, this won't re-instantiate complex objects.
            # if self.config['summary_creator_id'] == 'ExtractiveSummaryCreator':
            #    self.summary_creator = ExtractiveSummaryCreator(max_words=self.config.get('summary_creator_max_words', 25))
            pass


        self.update_summaries = bool(self.config.get('update_summaries', update_summaries))

        dedup_cache_size = int(self.config.get('dedup_cache_size', dedup_cache))
        self._dedup = _LRUSet(size=dedup_cache_size)

        # _chunker is already set by super() via self.chunker property of BaseCompressionEngine
        # self._chunker = self.chunker

        self.metrics = {
            "memories_ingested": 0,
            "prototypes_spawned": 0,
            "duplicates_skipped": 0,
            "prototypes_updated": 0,
            "prototype_vector_change_magnitude": 0.0,
        }
        self.prompt_budget = prompt_budget # Not yet managed by config
        # self.preprocess_fn is set by super()

    # ------------------------------------------------------------------
    @property
    def chunker(self) -> Chunker:
        """Return the current :class:`Chunker`."""
        return self._chunker

    @chunker.setter
    def chunker(self, value: Chunker) -> None:
        if not isinstance(value, Chunker):
            raise TypeError("chunker must implement Chunker interface")
        self._chunker = value
        self.store.meta["chunker"] = getattr(value, "id", type(value).__name__)

    # ------------------------------------------------------------------
    @property
    def similarity_threshold(self) -> float:
        return self._similarity_threshold

    @similarity_threshold.setter
    def similarity_threshold(self, value: float) -> None:
        self._similarity_threshold = float(value)
        self.store.meta["tau"] = float(value)

    # ------------------------------------------------------------------
    @property
    def summary_creator(self) -> MemoryCreator:
        return self._summary_creator

    @summary_creator.setter
    def summary_creator(self, value: MemoryCreator) -> None:
        self._summary_creator = value

    # ------------------------------------------------------------------
    def configure(
        self,
        *,
        chunker: Chunker | None = None,
        similarity_threshold: float | None = None,
        summary_creator: MemoryCreator | None = None,
        prompt_budget: PromptBudget | None = None,
    ) -> None:
        """Dynamically update core configuration."""

        if chunker is not None:
            self.chunker = chunker
        if similarity_threshold is not None:
            self.similarity_threshold = similarity_threshold
        if summary_creator is not None:
            self.summary_creator = summary_creator
        if prompt_budget is not None:
            self.prompt_budget = prompt_budget

    # ------------------------------------------------------------------
    def get_statistics(self) -> Dict[str, object]:
        """Return summary statistics about the current store."""

        return {
            "prototypes": len(self.store.prototypes),
            "memories": len(self.store.memories),
            "tau": self.similarity_threshold,
            "updated": self.store.meta.get("updated_at"),
        }

    # ------------------------------------------------------------------
    def _repr_html_(self) -> str:
        """HTML summary for notebooks."""
        stats = self.get_statistics()
        rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in stats.items())
        return f"<h3>PrototypeEngine</h3><table>{rows}</table>"

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
        source_document_id: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        """
        Ingests a piece of text into the engine store.

        The text is processed by the configured chunker, and each chunk is then
        added to the memory system. This may involve creating new prototypes or
        merging with existing ones based on similarity.

        Args:
            text: The text content to ingest.
            who, what, when, where, why: Optional metadata fields that can be associated
                                        with the ingested memory (currently passed to
                                        the underlying prototype system but full support
                                        may vary).
            progress_callback: An optional function that can be called to report
                               progress during the ingestion of multiple chunks.
                               The callback might receive (current_chunk, total_chunks,
                               is_new_prototype, status_message, similarity_score).
            source_document_id: An optional identifier for the source of the text
                                (e.g., a filename or URL).

        Returns:
            A list of dictionaries, where each dictionary represents the status or
            outcome of processing a single chunk from the input text.
        """
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in self._dedup:
            self.metrics["duplicates_skipped"] += 1
            return [{"duplicate": True}]
        self._dedup.add(digest)

        if self.preprocess_fn is not None:
            text = self.preprocess_fn(text)

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
        """
        Queries the engine store based on the input text.

        This method embeds the input text and searches for the most similar
        prototypes and individual memories in the store.

        Args:
            text: The query text.
            top_k_prototypes: The maximum number of most similar prototypes to return.
            top_k_memories: The maximum number of most similar individual memories
                            (associated with the top prototypes) to return.
            include_hypotheses: If True, may include hypothetical or inferred data
                                as part of the query process (specific behavior depends
                                on the underlying `PrototypeSystemEngine`).

        Returns:
            A `QueryResult` object (which is a dict subclass) containing:
                - "prototypes": A list of `PrototypeHit` dicts.
                - "memories": A list of `MemoryHit` dicts.
                - "status": A status message about the query.
        """
        vec = embed_text(text)
        if vec.ndim != 1:
            vec = vec.reshape(-1)
        nearest = self.store.find_nearest(vec, k=top_k_prototypes)
        if not nearest:
            return QueryResult([], [], "no_match")

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

        return QueryResult(proto_results, mem_results, "ok")

    # ------------------------------------------------------------------
    def compress(
        self,
        text_or_chunks: str | List[str],
        llm_token_budget: int,
        *,
        tokenizer=None,
    ) -> CompressedMemory:
        """Compress by retrieving relevant memories and truncating."""

        if isinstance(text_or_chunks, list):
            query_text = " ".join(text_or_chunks)
        else:
            query_text = text_or_chunks

        result = self.query(query_text, top_k_prototypes=1, top_k_memories=3)
        proto_summaries = "; ".join(p["summary"] for p in result.get("prototypes", []))
        mem_texts = "; ".join(m["text"] for m in result.get("memories", []))
        combined = " ".join(filter(None, [proto_summaries, mem_texts]))
        if tokenizer is not None:
            compressed = truncate_text(tokenizer, combined, llm_token_budget)
        else:
            compressed = combined[:llm_token_budget]
        return CompressedMemory(text=compressed, metadata={"status": result["status"]})

    # ------------------------------------------------------------------
    def process_conversational_turn(
        self,
        input_message: str,
        manager: ActiveMemoryManager,
        *,
        compression: BaseCompressionEngine | None = None,
    ) -> tuple[str, dict]:
        """Generate a reply using ``manager`` and optional ``compression``.

        ``manager`` controls which conversation history is supplied to the LLM. Applications may
        pass in a custom-configured instance to tailor recency and relevance behaviour.
        """

        from .local_llm import LocalChatModel

        llm = getattr(self, "_chat_model", None)
        if llm is None:
            llm = LocalChatModel()
            self._chat_model = llm

        if compression is None:
            compression = NoCompressionEngine()

        if getattr(llm, "tokenizer", None) is None:
            try:
                llm.load_model()
            except Exception:
                pass

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
        if compression is not None:
            limit = None
            if b_recent or b_older:
                limit = (b_recent or 0) + (b_older or 0)
            compressed, trace = compression.compress(
                [t.text for t in history_final],
                limit,
                tokenizer=llm.tokenizer,
            )
            history_text = compressed.text
            history_tokens_final = token_count(llm.tokenizer, history_text)

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
        prompt_tokens = after_llm_tokens
        reply = llm.reply(prompt)

        manager.add_turn(
            ConversationTurn(
                text=f"User: {input_message}\nAgent: {reply}",
                turn_embedding=vec.tolist() if hasattr(vec, "tolist") else None,
            )
        )
        return reply, {
            "query_result": query_res,
            "prompt_tokens": prompt_tokens,
            "compression_trace": trace if compression is not None else None,
        }

    # ------------------------------------------------------------------
    def receive_channel_message(
        self,
        source_id: str,
        message_text: str,
        manager: Optional[ActiveMemoryManager] = None,
        *,
        compression: BaseCompressionEngine | None = None,
    ) -> dict[str, object]:
        """
        Processes a message received from a channel or user.

        This method provides a high-level interface for the engine to react to
        incoming messages. The default behavior is:
        - If the message ends with "?", it's treated as a query. The engine
          will attempt to generate a response using its memory and an LLM
          (if available and configured).
        - Otherwise, the message is ingested as a new memory into the store.

        Developers can override or extend this method for more complex engine behaviors.

        Args:
            source_id: An identifier for the source of the message (e.g., user ID, channel name).
            message_text: The content of the message.
            manager: An `ActiveMemoryManager` instance to control which conversation
                     history is used when generating a response to a query. If None,
                     a simpler query without conversational history management is performed.
            compression: An optional :class:`BaseCompressionEngine` instance to
                compress the conversational history or context before sending it
                to an LLM. If ``None``, no compression is applied (or
                :class:`NoCompressionEngine` is used).

        Returns:
            A dictionary summarizing the action taken by the engine and any results.
            Common keys include:
                - "source": The `source_id`.
                - "action": "query" or "ingest".
                - "query_result": Result from `query()` if it was a query.
                - "reply": The LLM-generated reply string, if applicable.
                - "prompt_tokens": Number of tokens in the prompt sent to the LLM.
                - "compression_trace": `CompressionTrace` object if compression was used.
                - "chunks_ingested": Number of chunks ingested if it was an ingest action.
        """
        logging.info("[receive] from %s: %s", source_id, message_text[:40])

        summary: dict[str, object] = {"source": source_id}

        text = message_text.strip()
        if text.endswith("?"):
            # -------------------------------------------------- Option A: query
            if manager is not None:
                reply, info = self.process_conversational_turn(
                    text, manager, compression=compression
                )
                summary["action"] = "query"
                summary["query_result"] = info.get("query_result", {})
                summary["reply"] = reply
                summary["prompt_tokens"] = info.get("prompt_tokens")
                if info.get("compression_trace") is not None:
                    summary["compression_trace"] = info["compression_trace"]
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

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Persist engine state to ``path``."""
        from pathlib import Path
        import json

        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        # Update self.config with PrototypeEngine-specific serializable settings
        self.config['similarity_threshold'] = self.similarity_threshold
        self.config['dedup_cache_size'] = self._dedup.size
        self.config['update_summaries'] = self.update_summaries
        self.config['summary_creator_id'] = type(self.summary_creator).__name__
        # Example for summary_creator params (if simple like max_words)
        if isinstance(self.summary_creator, ExtractiveSummaryCreator):
            self.config['summary_creator_max_words'] = self.summary_creator.max_words

        self.config['store_id'] = type(self.store).__name__
        # If store has its own simple config (e.g. embedding_dim was crucial for init)
        # self.config['store_embedding_dim'] = self.store.embedding_dim # Assuming store has this
        # self.store.meta['tau'] is effectively self.similarity_threshold, already saved.
        # Other parts of self.store.meta are saved below.

        super().save(p) # This saves engine_manifest.json with self.config

        # PrototypeEngine specific save logic
        # It re-reads and updates the manifest written by super().save()
        # This is acceptable if it's adding top-level keys not part of 'config'
        with open(p / "engine_manifest.json", "r", encoding="utf-8") as fh:
            manifest = json.load(fh)

        manifest.update(
            {
                "meta": self.store.meta, # Saves tau and other store metadata
                "prototypes": [
                    p.model_dump(mode="json") for p in self.store.prototypes
                ],
            }
        )
        with open(p / "engine_manifest.json", "w", encoding="utf-8") as fh:
            json.dump(manifest, fh)

        with open(p / "memories.json", "w", encoding="utf-8") as fh:
            json.dump([m.model_dump(mode="json") for m in self.store.memories], fh)

        import numpy as np
        np.save(p / "vectors.npy", self.store.proto_vectors)

    # ------------------------------------------------------------------
    def load(self, path: str | Path) -> None:
        """Load engine state from ``path``."""
        from pathlib import Path
        import json
        import numpy as np
        from .models import BeliefPrototype, RawMemory
        from compact_memory.vector_store import InMemoryVectorStore # For type checking or re-init
        from compact_memory.embedding_pipeline import get_embedding_dim


        p = Path(path)

        # self.config should already be populated by __init__ via load_engine.
        # Attributes like self.similarity_threshold, self._dedup.size, self.update_summaries
        # should have been initialized from self.config in __init__.

        # Load the manifest to get prototype-specific data
        with open(p / "engine_manifest.json", "r", encoding="utf-8") as fh:
            manifest = json.load(fh)

        # Load memories
        with open(p / "memories.json", "r", encoding="utf-8") as fh:
            memories_data = json.load(fh)

        # Ensure self.store is initialized (should be by __init__)
        # If it's a type that needs specific re-initialization based on manifest['meta']
        # (e.g. if embedding_dim changed or store type changed), handle here.
        # For InMemoryVectorStore, __init__ already created it. We just populate it.

        # Re-populate the store
        self.store.meta = manifest.get("meta", {}) # meta includes tau (similarity_threshold)

        # If similarity_threshold from config differs from store.meta['tau'], reconcile
        # Typically, store.meta['tau'] (loaded from manifest['meta']) should be source of truth for store's behavior.
        # And self.similarity_threshold (from config) should align with it.
        # The property setter for self.similarity_threshold already updates self.store.meta['tau'].
        # So, if config had similarity_threshold, it would have been set in __init__, updating store.meta.
        # Then, manifest['meta'] loaded here might overwrite it. Let's ensure consistency:
        # self.similarity_threshold setter will make them consistent if called.
        # Or, prioritize manifest['meta']['tau'] if present.
        if 'tau' in self.store.meta:
            self.similarity_threshold = float(self.store.meta['tau'])
        else: # Ensure tau is in meta if not loaded but set from config
            self.store.meta['tau'] = self.similarity_threshold


        self.store.prototypes = [
            BeliefPrototype(**p) for p in manifest.get("prototypes", [])
        ]
        self.store.proto_vectors = np.load(p / "vectors.npy")
        self.store.memories = [RawMemory(**m) for m in memories_data]

        # Rebuild the FAISS index if it's an InMemoryVectorStore with FAISS
        if hasattr(self.store, 'create_index'):
             self.store.create_index() # For InMemoryVectorStore, this rebuilds FAISS
        else: # For basic VectorStore or if index is managed differently
            self.store.index = {
                p.prototype_id: i for i, p in enumerate(self.store.prototypes)
            }
            self.store._index_dirty = True

        # Note: super().load(path) is NOT called as PrototypeEngine manages its own state files
        # distinct from BaseCompressionEngine's (entries.json, embeddings.npy).
