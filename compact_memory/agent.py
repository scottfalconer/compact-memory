from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, TypedDict, Callable

import logging


from .chunker import Chunker, SentenceWindowChunker
from .embedding_pipeline import embed_text
from .vector_store import VectorStore
from .memory_creation import (
    ExtractiveSummaryCreator,
    MemoryCreator,
)
from .prompt_budget import PromptBudget
from .token_utils import truncate_text, token_count
from .strategies.experimental import (
    ActiveMemoryManager,
    ConversationTurn,
    PrototypeSystemStrategy,
)
from .compression.strategies_abc import CompressedMemory, CompressionTrace
from .compression import CompressionStrategy, NoCompression


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
    """Result object returned by :meth:`Agent.query` with HTML representation."""

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


class Agent:
    """
    Core component for managing and interacting with a memory store.

    The Agent class encapsulates the logic for ingesting text into a
    `VectorStore`, managing memory prototypes, querying the memory,
    and processing conversational turns with optional compression.

    It utilizes a `PrototypeSystemStrategy` internally to handle the
    mechanics of memory consolidation and retrieval. Developers typically
    interact with the Agent by providing it with a pre-configured store
    and then using its methods like `add_memory`, `query`, and
    `receive_channel_message`.

    Attributes:
        store (VectorStore): The underlying vector store for memories and prototypes.
        prototype_system (PrototypeSystemStrategy): Handles memory consolidation and querying logic.
        metrics (Dict[str, Any]): A dictionary of metrics collected during operations
                                  (primarily from `PrototypeSystemStrategy`).
        prompt_budget (Optional[PromptBudget]): Configuration for managing prompt sizes
                                             when interacting with LLMs.
    """

    def __init__(
        self,
        store: VectorStore,
        *,
        chunker: Optional[Chunker] = None,
        similarity_threshold: float = 0.8,
        dedup_cache: int = 128,
        summary_creator: Optional[MemoryCreator] = None,
        update_summaries: bool = False,
        prompt_budget: PromptBudget | None = None,
        preprocess_fn: Callable[[str], str] | None = None,
    ) -> None:
        """
        Initializes the Agent.

        Args:
            store: The `VectorStore` instance to be used for storing
                   and retrieving memories and prototypes.
            chunker: A `Chunker` instance for splitting text during ingestion.
                     Defaults to `SentenceWindowChunker` if None.
            similarity_threshold: The threshold (tau) for determining if a new memory
                                  should be merged with an existing prototype.
                                  Value between 0.0 and 1.0.
            dedup_cache: The size of the cache used for deduplicating recent memories
                         to avoid redundant processing.
            summary_creator: A `MemoryCreator` instance responsible for generating
                             summaries for new prototypes. Defaults to
                             `ExtractiveSummaryCreator` if None.
            update_summaries: If True, summaries of existing prototypes may be updated
                              when new, highly similar memories are ingested.
            prompt_budget: A `PromptBudget` object to manage and allocate token budgets
                           for different parts of an LLM prompt (e.g., query, history, LTM).
        """
        self.store = store
        self.prototype_system = PrototypeSystemStrategy(
            store,
            chunker=chunker,
            similarity_threshold=similarity_threshold,
            dedup_cache=dedup_cache,
            summary_creator=summary_creator,
            update_summaries=update_summaries,
            preprocess_fn=preprocess_fn,
        )
        self.metrics = self.prototype_system.metrics
        self.prompt_budget = prompt_budget

    # ------------------------------------------------------------------
    @property
    def chunker(self) -> Chunker:
        """Return the current :class:`Chunker`."""
        return self.prototype_system.chunker

    @chunker.setter
    def chunker(self, value: Chunker) -> None:
        if not isinstance(value, Chunker):
            raise TypeError("chunker must implement Chunker interface")
        self.prototype_system.chunker = value
        self.store.meta["chunker"] = getattr(value, "id", type(value).__name__)

    # ------------------------------------------------------------------
    @property
    def similarity_threshold(self) -> float:
        return self.prototype_system.similarity_threshold

    @similarity_threshold.setter
    def similarity_threshold(self, value: float) -> None:
        self.prototype_system.similarity_threshold = value
        self.store.meta["tau"] = float(value)

    # ------------------------------------------------------------------
    @property
    def summary_creator(self) -> MemoryCreator:
        return self.prototype_system.summary_creator

    @summary_creator.setter
    def summary_creator(self, value: MemoryCreator) -> None:
        self.prototype_system.summary_creator = value

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

        from .utils import get_disk_usage

        path_value = self.store.path
        path = Path(path_value) if path_value else None
        disk_usage = get_disk_usage(path) if path is not None else 0
        return {
            "prototypes": len(self.store.prototypes),
            "memories": len(self.store.memories),
            "tau": self.similarity_threshold,
            "updated": self.store.meta.get("updated_at"),
            "disk_usage": disk_usage,
        }

    # ------------------------------------------------------------------
    def _repr_html_(self) -> str:
        """HTML summary for notebooks."""
        stats = self.get_statistics()
        rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in stats.items())
        return f"<h3>Agent</h3><table>{rows}</table>"

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
        save: bool = True,
        source_document_id: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        """
        Ingests a piece of text into the agent's memory store.

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
            save: If True (default), the memory store is saved to disk after ingestion.
            source_document_id: An optional identifier for the source of the text
                                (e.g., a filename or URL).

        Returns:
            A list of dictionaries, where each dictionary represents the status or
            outcome of processing a single chunk from the input text.
        """
        return self.prototype_system.add_memory(
            text,
            who=who,
            what=what,
            when=when,
            where=where,
            why=why,
            progress_callback=progress_callback,
            save=save,
            source_document_id=source_document_id,
        )

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
        Queries the agent's memory store based on the input text.

        This method embeds the input text and searches for the most similar
        prototypes and individual memories in the store.

        Args:
            text: The query text.
            top_k_prototypes: The maximum number of most similar prototypes to return.
            top_k_memories: The maximum number of most similar individual memories
                            (associated with the top prototypes) to return.
            include_hypotheses: If True, may include hypothetical or inferred data
                                as part of the query process (specific behavior depends
                                on the underlying `PrototypeSystemStrategy`).

        Returns:
            A `QueryResult` object (which is a dict subclass) containing:
                - "prototypes": A list of `PrototypeHit` dicts.
                - "memories": A list of `MemoryHit` dicts.
                - "status": A status message about the query.
        """
        res = self.prototype_system.query(
            text,
            top_k_prototypes=top_k_prototypes,
            top_k_memories=top_k_memories,
            include_hypotheses=include_hypotheses,
        )
        return QueryResult(res["prototypes"], res["memories"], res["status"])

    # ------------------------------------------------------------------
    def process_conversational_turn(
        self,
        input_message: str,
        manager: ActiveMemoryManager,
        *,
        compression: CompressionStrategy | None = None,
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
            compression = NoCompression()

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
        compression: CompressionStrategy | None = None,
    ) -> dict[str, object]:
        """
        Processes a message received from a channel or user.

        This method provides a high-level interface for an agent to react to
        incoming messages. The default behavior is:
        - If the message ends with "?", it's treated as a query. The agent
          will attempt to generate a response using its memory and an LLM
          (if available and configured).
        - Otherwise, the message is ingested as a new memory into the store.

        Developers can override or extend this method for more complex agent behaviors.

        Args:
            source_id: An identifier for the source of the message (e.g., user ID, channel name).
            message_text: The content of the message.
            manager: An `ActiveMemoryManager` instance to control which conversation
                     history is used when generating a response to a query. If None,
                     a simpler query without conversational history management is performed.
            compression: An optional `CompressionStrategy` instance to compress
                         the conversational history or context before sending it to an LLM.
                         If None, no compression is applied (or `NoCompression` strategy is used).

        Returns:
            A dictionary summarizing the action taken by the agent and any results.
            Common keys include:
                - "source": The `source_id`.
                - "action": "query" or "ingest".
                - "query_result": Result from `agent.query()` if it was a query.
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
