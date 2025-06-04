from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, TypedDict, Callable

import logging


from .chunking import ChunkFn
from .memory_store import MemoryStore
from .memory_creation import ExtractiveSummaryCreator, MemoryCreator
from .prompt_budget import PromptBudget
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
    `MemoryStore`, managing memory prototypes, and querying the memory.

    It utilizes a `PrototypeSystemStrategy` internally to handle the
    mechanics of memory consolidation and retrieval. Developers typically
    interact with the Agent by providing it with a pre-configured store
    and then using its methods like `add_memory` and `query`.

    Attributes:
        store (MemoryStore): The underlying memory store for prototypes and vectors.
        prototype_system (PrototypeSystemStrategy): Handles memory consolidation and querying logic.
        metrics (Dict[str, Any]): A dictionary of metrics collected during operations
                                  (primarily from `PrototypeSystemStrategy`).
        prompt_budget (Optional[PromptBudget]): Configuration for managing prompt sizes
                                             when interacting with LLMs.
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        chunk_fn: ChunkFn | None = None,
        similarity_threshold: float = 0.8,
        dedup_cache: int = 128,
        summary_creator: Optional[MemoryCreator] = None,
        update_summaries: bool = False,
        prompt_budget: PromptBudget | None = None,
    ) -> None:
        """
        Initializes the Agent.

        Args:
            store: The :class:`MemoryStore` instance used for managing
                prototypes and vectors.
            chunk_fn: Optional function used to split text into chunks during
                ingestion. If ``None``, the entire text is treated as a single
                chunk.
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
            chunk_fn=chunk_fn,
            similarity_threshold=similarity_threshold,
            dedup_cache=dedup_cache,
            summary_creator=summary_creator,
            update_summaries=update_summaries,
        )
        self.metrics = self.prototype_system.metrics
        self.prompt_budget = prompt_budget

    # ------------------------------------------------------------------
    @property
    def chunk_fn(self) -> ChunkFn | None:
        """Return the current chunking function."""
        return self.prototype_system.chunk_fn

    @chunk_fn.setter
    def chunk_fn(self, value: ChunkFn | None) -> None:
        self.prototype_system.chunk_fn = value

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
        chunk_fn: ChunkFn | None = None,
        similarity_threshold: float | None = None,
        summary_creator: MemoryCreator | None = None,
        prompt_budget: PromptBudget | None = None,
    ) -> None:
        """Dynamically update core configuration."""

        if chunk_fn is not None:
            self.chunk_fn = chunk_fn
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

        The text is processed by the configured ``chunk_fn``. Each resulting
        chunk is then
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
