from __future__ import annotations

import os # For persistence paths
from pathlib import Path
import json # For agent config persistence
from typing import Dict, List, Optional, TypedDict, Callable, Type # Added Type for class method loading

import logging


from .chunker import Chunker, SentenceWindowChunker
from .embedding_pipeline import embed_text, EmbeddingFunction # Import EmbeddingFunction
from .vector_store import BaseVectorStore, InMemoryVectorStore
from .models import BeliefPrototype, ConversationTurn # Added ConversationTurn from models
from .memory_creation import (
    ExtractiveSummaryCreator,
    MemoryCreator,
)
from .prompt_budget import PromptBudget
from .token_utils import truncate_text, token_count
from .active_memory_manager import ActiveMemoryManager # ConversationTurn removed as it's from models
from .compression.strategies_abc import CompressedMemory, CompressionTrace
from .compression import CompressionStrategy, NoCompression
from .prototype_system_strategy import PrototypeSystemStrategy


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
    `InMemoryVectorStore`, managing memory prototypes, querying the memory,
    and processing conversational turns with optional compression.

    It utilizes a `PrototypeSystemStrategy` internally to handle the
    mechanics of memory consolidation and retrieval. Developers typically
    interact with the Agent by providing it with a pre-configured store
    and then using its methods like `add_memory`, `query`, and
    `receive_channel_message`.

    Attributes:
        vector_store (BaseVectorStore): The underlying vector store.
        prototype_system (PrototypeSystemStrategy): Handles memory consolidation and querying logic.
        active_memory_manager (ActiveMemoryManager): Manages conversational history.
        embedding_dim (int): Dimension of embeddings used.
        metrics (Dict[str, Any]): A dictionary of metrics collected during operations.
        prompt_budget (Optional[PromptBudget]): Configuration for managing prompt sizes.
    """
    DEFAULT_EMBEDDING_DIM = 384 # Example default

    def __init__(
        self,
        vector_store: Optional[BaseVectorStore] = None,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        embedding_fn: Optional[EmbeddingFunction] = None, # Added embedding_fn
        *,
        chunker: Optional[Chunker] = None,
        similarity_threshold: float = 0.8,
        dedup_cache: int = 128,
        summary_creator: Optional[MemoryCreator] = None,
        update_summaries: bool = False,
        prompt_budget: PromptBudget | None = None,
        amm_config: Optional[Dict] = None,
        store_turn_embeddings_in_amm: bool = False,
    ) -> None:
        """
        Initializes the Agent.

        Args:
            vector_store: An instance of BaseVectorStore. If None, InMemoryVectorStore is used.
            embedding_dim: Dimension of the embeddings. Crucial if using custom embedding_fn.
            embedding_fn: Optional custom function for generating embeddings.
            chunker: Chunker for splitting text. Defaults to SentenceWindowChunker.
            similarity_threshold: Threshold for merging memories with prototypes.
            dedup_cache: Size of deduplication cache.
            summary_creator: For generating prototype summaries. Defaults to ExtractiveSummaryCreator.
            update_summaries: Whether to update summaries of existing prototypes.
            prompt_budget: PromptBudget object for LLM interactions.
            amm_config: Configuration dictionary for ActiveMemoryManager.
            store_turn_embeddings_in_amm: Whether AMM should store turn embeddings in its vector_store.
        """
        self.embedding_dim = embedding_dim
        self.embedding_fn = embedding_fn # Store embedding function

        if vector_store is None:
            self.vector_store: BaseVectorStore = InMemoryVectorStore(embedding_dim=self.embedding_dim)
        else:
            self.vector_store = vector_store

        self.prototype_system = PrototypeSystemStrategy(
            self.vector_store,
            embedding_dim=self.embedding_dim,
            embedding_fn=self.embedding_fn, # Pass embedding_fn to PSS
            chunker=chunker,
            similarity_threshold=similarity_threshold,
            dedup_cache=dedup_cache,
            summary_creator=summary_creator,
            update_summaries=update_summaries,
        )
        self.metrics = self.prototype_system.metrics
        self.prompt_budget = prompt_budget

        amm_config_values = amm_config or {}
        self.active_memory_manager = ActiveMemoryManager(
            vector_store=self.vector_store, # AMM can use the same vs, or a different one if needed
            embedding_dim=self.embedding_dim,
            store_turn_embeddings=store_turn_embeddings_in_amm,
            **amm_config_values
        )

    # ------------------------------------------------------------------
    @property
    def chunker(self) -> Chunker: # type: ignore
        """Return the current :class:`Chunker`."""
        return self.prototype_system.chunker

    @chunker.setter
    def chunker(self, value: Chunker) -> None: # type: ignore
        if not isinstance(value, Chunker):
            raise TypeError("chunker must implement Chunker interface")
        self.prototype_system.chunker = value
        # self.vector_store.meta is not a thing in BaseVectorStore. Metadata needs careful handling.
        # If chunker type needs to be persisted, Agent's save_agent should handle it.
        # For now, removing direct write to store.meta here.
        # self.store.meta["chunker"] = getattr(value, "id", type(value).__name__)

    # ------------------------------------------------------------------
    @property
    def similarity_threshold(self) -> float: # type: ignore
        return self.prototype_system.similarity_threshold

    @similarity_threshold.setter
    def similarity_threshold(self, value: float) -> None: # type: ignore
        self.prototype_system.similarity_threshold = value
        # self.store.meta["tau"] = float(value) # Same as above, Agent should persist this.

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
        # Removed disk_usage as it's not general for BaseVectorStore
        # "updated_at" was from store.meta, also not general.
        # These could be added if a specific store instance provides them.
        return {
            "prototypes": len(self.prototype_system.prototypes), # Get from PSS
            "memories": len(self.prototype_system.memories),   # Get from PSS
            "tau": self.similarity_threshold,
            # Add other relevant stats if available
        }

    # ------------------------------------------------------------------
    def _repr_html_(self) -> str:
        """HTML summary for notebooks."""
        stats = self.get_statistics()
        rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in stats.items())
        return f"<h3>Agent</h3><table>{rows}</table>"

    # ------------------------------------------------------------------
    def get_prototypes_view( # type: ignore
        self, sort_by: str | None = None
    ) -> List[Dict[str, object]]:
        """Return list of prototypes for display purposes."""

        # Prototypes are now stored in PrototypeSystemStrategy
        protos = list(self.prototype_system.prototypes.values())
        if sort_by == "strength":
            protos.sort(key=lambda p: p.strength, reverse=True)
        # TODO: Ensure BeliefPrototype model is imported or structure is compatible
        return [
            {
                "id": p.prototype_id, # Make sure these attributes exist on BeliefPrototype
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
        # `save` parameter removed from PSS.add_memory as Agent handles persistence.
        return self.prototype_system.add_memory(
            text,
            who=who,
            what=what,
            when=when,
            where=where,
            why=why,
            progress_callback=progress_callback,
            # save=save, # PSS no longer handles saving the main vector store
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

        # vec generation should use self.embedding_dim and self.embedding_fn
        input_embedding = embed_text([input_message], embedding_fn=self.embedding_fn)
        if input_embedding.ndim != 1: # Should be (1, dim) or (dim,)
            # If embed_text for single item returns (1, dim), take first row.
            # If it's already (dim,), this is fine.
            if input_embedding.shape[0] == 1:
                 input_embedding = input_embedding.reshape(-1)
            else: # Should not happen with current embed_text logic for single string
                logging.warning(f"Unexpected embedding shape for single message: {input_embedding.shape}")
                # Attempt to take the first vector if it's multi-vector for some reason
                if input_embedding.shape[0] > 0:
                    input_embedding = input_embedding[0].reshape(-1)
                else: # Fallback to zeros if shape is totally unexpected (e.g. (0,dim))
                    input_embedding = np.zeros(self.embedding_dim, dtype=np.float32)


        history_candidates = manager.select_history_candidates_for_prompt(input_embedding) # AMM uses this embedding
        # ConversationTurn text attribute might need adjustment if it's a Pydantic model from .models
        # The current ConversationTurn in AMM was a simple dataclass.
        # Now AMM uses ConversationTurn from models.py which has user_message, agent_response

        # Construct full text for token counting if using models.ConversationTurn
        history_texts = [f"{t.user_message}\n{t.agent_response}".strip() for t in history_candidates]
        candidate_tokens = token_count(
            llm.tokenizer, "\n".join(history_texts)
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
        history_final = older_final + recent_final # This is List[ConversationTurn]

        # Construct history_text from user_message and agent_response
        final_history_texts = [f"{t.user_message}\n{t.agent_response}".strip() for t in history_final]
        history_text = "\n".join(final_history_texts)
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
                final_history_texts, # Pass list of strings
                limit,
                tokenizer=llm.tokenizer,
            )
            history_text = compressed.text # compressed.text is the compressed string
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

        # AMM.add_turn expects a ConversationTurn object from models.py
        # The ConversationTurn model has user_message and agent_response fields.
        current_turn = ConversationTurn(
            user_message=input_message, # Store raw input message
            agent_response=reply,
            turn_embedding=input_embedding.tolist() if hasattr(input_embedding, "tolist") else None,
            # turn_id will be auto-generated by Pydantic model
        )
        manager.add_turn(current_turn)

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
        # The PSS.add_memory no longer takes `save` argument.
        results = self.add_memory(
            text,
            source_document_id=f"session_post_from:{source_id}",
        )
        # Agent-level persistence call after ingestion if needed
        # self.vector_store.persist() # Or self.save_agent(path_to_agent_data)
        # This depends on how frequently persistence should occur.
        # For now, persistence is explicit via save_agent.

        summary.update(
            {
                "action": "ingest",
                "chunks_ingested": len(results),
                "reply": None,
            }
        )
        return summary

    # ------------------------------------------------------------------
    # Persistence methods
    # ------------------------------------------------------------------
    def save_agent(self, agent_dir_path: str) -> None:
        """Saves the agent's state to the specified directory."""
        p_path = Path(agent_dir_path)
        p_path.mkdir(parents=True, exist_ok=True)

        # Save Agent's configuration
        # Prepare vector store config
        vs_config = {}
        vs_type_name = type(self.vector_store).__name__
        if vs_type_name == "ChromaVectorStoreAdapter":
            # Assuming ChromaVectorStoreAdapter has 'path' and 'collection_name' attributes
            vs_config["path"] = getattr(self.vector_store, 'path', None)
            vs_config["collection_name"] = getattr(self.vector_store, 'collection_name', "compact_memory_default")
        elif vs_type_name == "FaissVectorStoreAdapter":
            # Faiss path is handled by where its files are saved (e.g., relative to agent_dir_path/vector_store)
            # No specific path config needed here if it always saves to a standard sub-path.
            # If its source index file path was important for re-loading an *external* index, store it.
            pass # Path is implicit for save/load relative to agent_dir_path

        # Prepare embedding function config
        # We primarily store info about the default Hugging Face provider if used.
        # Custom functions are identified by "embedding_fn_configured": True
        embedding_provider_type = "custom" if self.embedding_fn else "huggingface"
        embedding_model_name = None
        embedding_device = None
        if not self.embedding_fn: # Using default HF pipeline
            # These would need to be accessible, e.g. if Agent stores them
            # For now, assume these are not explicitly stored on Agent if using default pipeline.
            # The CLI or user would know these if they overrode defaults for the default pipeline.
            # Let's assume for saving, we don't have these explicitly if self.embedding_fn is None.
            # The CLI helper will use its own defaults (HF_DEFAULT_MODEL_NAME) if not in config.
            pass


        agent_config = {
            "embedding_dim": self.embedding_dim,
            "similarity_threshold": self.prototype_system.similarity_threshold,
            "dedup_cache_size": self.prototype_system._dedup.size,
            "update_summaries": self.prototype_system.update_summaries,
            "chunker_config": self.chunker.to_dict() if hasattr(self.chunker, 'to_dict') else type(self.chunker).__name__,
            "vector_store_type": vs_type_name,
            "vector_store_config": vs_config,
            "embedding_provider_type": embedding_provider_type,
            "embedding_model_name": embedding_model_name, # Will be None if custom_fn or default not storing these
            "embedding_device": embedding_device,       # Will be None if custom_fn or default not storing these
            "embedding_fn_configured": self.embedding_fn is not None,
            # TODO: Persist amm_config, store_turn_embeddings_in_amm for AMM
            # TODO: Persist prompt_budget
        }
        with open(p_path / "agent_config.json", "w") as f:
            json.dump(agent_config, f, indent=4)

        # Persist the vector store
        # Some stores might be no-op (InMemory), others (Faiss, Chroma) will save.
        # We need a dedicated sub-directory for the vector store data.
        vs_path = p_path / "vector_store"
        vs_path.mkdir(parents=True, exist_ok=True)
        if hasattr(self.vector_store, 'persist') and callable(getattr(self.vector_store, 'persist')):
            # FaissAdapter expects a dir path, InMemory is no-op, Chroma might be no-op or take path
            # This requires persist methods of different adapters to be consistent or handled here.
            # For FaissAdapter, it saves multiple files IN the given path.
            # For Chroma, if persistent, it's handled by Chroma client.
            # Let's assume persist() is smart enough or takes a path where it can save its specific files.
            # The Faiss adapter saves to a directory given to its persist method.
            try:
                self.vector_store.persist(str(vs_path)) # Pass string path
            except NotImplementedError:
                logging.warning(f"Vector store {type(self.vector_store).__name__} does not implement persist.")
            except Exception as e:
                logging.error(f"Error persisting vector store: {e}")


        # Persist PrototypeSystemStrategy state (prototypes and memories dicts)
        pss_path = p_path / "prototype_system_state"
        self.prototype_system.save_state(pss_path)

        # TODO: Persist ActiveMemoryManager state (e.g., history if needed, though usually ephemeral)

        logging.info(f"Agent state saved to {agent_dir_path}")

    @classmethod
    def load_agent(
        cls,
        agent_dir_path: str,
        vector_store_instance: BaseVectorStore,
        embedding_fn: Optional[EmbeddingFunction] = None, # Allow passing embedding_fn at load time
        # Optional: pass specific class types for chunker, summary_creator if not reconstructible from config
        chunker_class: Optional[Type[Chunker]] = None,
        summary_creator_class: Optional[Type[MemoryCreator]] = None
    ) -> "Agent":
        """Loads the agent's state from the specified directory."""
        p_path = Path(agent_dir_path)
        if not p_path.exists() or not p_path.is_dir():
            raise FileNotFoundError(f"Agent directory {agent_dir_path} not found.")

        with open(p_path / "agent_config.json", "r") as f:
            agent_config = json.load(f)

        embedding_dim = agent_config.get("embedding_dim", cls.DEFAULT_EMBEDDING_DIM)

        if agent_config.get("embedding_fn_configured") and embedding_fn is None:
            logging.warning("Agent was saved with a custom embedding_fn, but none provided at load time. Defaulting.")

        # Load the vector store state
        vs_path = p_path / "vector_store"
        if vs_path.exists() and hasattr(vector_store_instance, 'load') and callable(getattr(vector_store_instance, 'load')):
            try:
                vector_store_instance.load(str(vs_path)) # Pass string path
            except NotImplementedError:
                logging.warning(f"Vector store {type(vector_store_instance).__name__} does not implement load or data not found.")
            except Exception as e:
                logging.error(f"Error loading vector store: {e}")
        elif vs_path.exists():
             logging.warning(f"Vector store at {vs_path} exists but instance {type(vector_store_instance).__name__} has no load method or it failed.")


        # TODO: Reconstruct chunker and summary_creator from config
        # This is simplified; robust reconstruction might need class registry or explicit passing
        chunker_name = agent_config.get("chunker_config", "SentenceWindowChunker")
        loaded_chunker = None
        if chunker_class and (chunker_class.__name__ == chunker_name or isinstance(chunker_name, dict)):
             loaded_chunker = chunker_class() # Add from_dict if config is dict
        elif chunker_name == "SentenceWindowChunker":
             loaded_chunker = SentenceWindowChunker()
        # Add more chunker types as needed or use a factory

        # For summary_creator, similar logic (simplified)
        loaded_summary_creator = ExtractiveSummaryCreator(max_words=25) # Default

        # Create Agent instance (or a new constructor for loading)
        # For now, use __init__ and then load PSS state
        agent = cls(
            vector_store=vector_store_instance,
            embedding_dim=embedding_dim,
            embedding_fn=embedding_fn, # Pass loaded/provided embedding_fn
            chunker=loaded_chunker,
            similarity_threshold=agent_config.get("similarity_threshold", 0.8),
            dedup_cache=agent_config.get("dedup_cache_size", 128), # Load dedup_cache for PSS
            summary_creator=loaded_summary_creator,
            update_summaries=agent_config.get("update_summaries", False),
            # TODO: Load prompt_budget, amm_config for Agent and AMM
        )

        # Load PrototypeSystemStrategy state
        pss_path = p_path / "prototype_system_state"
        if pss_path.exists():
            agent.prototype_system.load_state(pss_path)
        else:
            logging.warning(f"PrototypeSystemStrategy state not found at {pss_path}")

        # TODO: Load ActiveMemoryManager state if persisted

        logging.info(f"Agent state loaded from {agent_dir_path}")
        return agent
