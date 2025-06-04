import logging
from typing import Dict, List, Any, Optional, Union, Tuple

try:
    import numpy as np
except ImportError:
    np = None

from compact_memory.models import RawMemory, Message
from compact_memory.compression.strategies_abc import CompressionStrategy, CompressedMemory
from compact_memory.compression.trace import CompressionTrace
from compact_memory.token_utils import Tokenizer, get_tokenizer
from compact_memory.embedding_pipeline import EmbeddingPipeline


class LPCSelectorStrategy(CompressionStrategy):
    id = "lpc_selector_strategy"

    def __init__(
        self,
        parameters: Dict[str, Any],
        tokenizer: Optional[Tokenizer] = None,
        embedding_pipeline: Optional[EmbeddingPipeline] = None,
        llm_provider: Optional[Any] = None,
    ):
        super().__init__(parameters, tokenizer, embedding_pipeline, llm_provider)
        self.logger = logging.getLogger(__name__)

        lpc_metadata_filter = parameters.get("lpc_metadata_filter")
        if not isinstance(lpc_metadata_filter, dict):
            raise ValueError("lpc_metadata_filter must be a dictionary.")
        self.lpc_metadata_filter: Dict[str, Any] = lpc_metadata_filter

        self.max_lpcs_to_select: int = parameters.get("max_lpcs_to_select", 5)
        self.relevance_score_threshold: float = parameters.get(
            "relevance_score_threshold", 0.0
        )
        self.lpc_join_separator: str = parameters.get(
            "lpc_join_separator", "\n\n---\n\n"
        )

        if tokenizer is None:
            self.logger.warning("Tokenizer not provided. Using default tokenizer.")
            self.tokenizer: Tokenizer = get_tokenizer()
        else:
            self.tokenizer: Tokenizer = tokenizer

        self.embedding_pipeline: Optional[EmbeddingPipeline] = embedding_pipeline
        if self.embedding_pipeline:
            self.logger.info("Embedding pipeline provided.")
        else:
            self.logger.info("No embedding pipeline provided.")

    def _is_lpc(self, memory: RawMemory) -> bool:
        if not hasattr(memory, "metadata") or not memory.metadata:
            return False

        for key, value in self.lpc_metadata_filter.items():
            if key not in memory.metadata or memory.metadata[key] != value:
                return False
        return True

    def _score_lpc_relevance(
        self,
        lpc: RawMemory,
        context_query_text: str,
        context_query_embedding: Optional[List[float]],
    ) -> float:
        if (
            self.embedding_pipeline
            and context_query_embedding is not None
            and hasattr(lpc, "embedding")
            and lpc.embedding is not None
        ):
            if np is None:
                self.logger.warning(
                    "Numpy not available, falling back to basic keyword matching for LPC scoring."
                )
            else:
                try:
                    lpc_emb = np.array(lpc.embedding, dtype=float)
                    query_emb = np.array(context_query_embedding, dtype=float)

                    if lpc_emb.shape != query_emb.shape:
                        self.logger.warning(
                            f"Embedding shapes differ: LPC {lpc_emb.shape}, Query {query_emb.shape}. Returning 0.0 score."
                        )
                        return 0.0

                    norm_lpc = np.linalg.norm(lpc_emb)
                    norm_query = np.linalg.norm(query_emb)

                    if norm_lpc == 0 or norm_query == 0:
                        self.logger.warning(
                            "LPC or query embedding has zero norm. Returning 0.0 score."
                        )
                        return 0.0

                    similarity = np.dot(lpc_emb, query_emb) / (norm_lpc * norm_query)
                    return float(similarity)
                except Exception as e:
                    self.logger.error(f"Error calculating semantic similarity: {e}. Falling back to keyword matching.")
                    # Fall through to keyword matching

        # Fallback Scoring Path (Basic Keyword Matching)
        score = 0.0
        lpc_text_lower = lpc.text.lower()
        query_words = set(context_query_text.lower().split())

        if not query_words:
            return 0.0

        for word in query_words:
            if word in lpc_text_lower:
                score += 1.0

        return score / (len(query_words) + 1e-6)

    def _create_trace(
        self, original_count: int, final_count: int, details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return {
            "strategy_class": self.__class__.__name__,
            "parameters": self._parameters,
            "original_item_count": original_count,
            "final_item_count": final_count,
            "details": details or {},
        }

    def compress(
        self,
        memories_or_messages: List[Union[RawMemory, Message]],
        budget: int, # Max tokens for the selected LPCs
        context_query: Optional[Any] = None,
    ) -> Tuple[CompressedMemory, CompressionTrace]:
        context_query_text: str = ""
        context_query_embedding: Optional[List[float]] = None

        if context_query is None:
            self.logger.warning("No context_query provided for LPC selection.")
        elif isinstance(context_query, str):
            context_query_text = context_query
            if self.embedding_pipeline:
                try:
                    context_query_embedding = self.embedding_pipeline.embed_query(
                        context_query_text
                    )
                except Exception as e:
                    self.logger.error(f"Failed to embed context query string: {e}")
        elif hasattr(context_query, "text"):
            context_query_text = context_query.text
            if self.embedding_pipeline:
                context_query_embedding = getattr(context_query, "embedding", None)
                if context_query_embedding is None:
                    try:
                        context_query_embedding = self.embedding_pipeline.embed_query(
                            context_query_text
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Failed to embed context query object's text: {e}"
                        )
        else:
            self.logger.error(
                f"Unsupported context_query type: {type(context_query)}. Cannot extract text or embedding."
            )
            trace = self._create_trace(
                original_count=len(memories_or_messages),
                final_count=0,
                details={
                    "error": "Unsupported context_query type",
                    "query_type": str(type(context_query)),
                },
            )
            return CompressedMemory(text="", metadata={}), trace

        # Step 1: Filter for LPCs
        potential_lpcs: List[RawMemory] = []
        for item in memories_or_messages:
            if isinstance(item, RawMemory) and self._is_lpc(item):
                potential_lpcs.append(item)

        if not potential_lpcs:
            self.logger.info("No potential LPCs found in the provided memories.")
            trace = self._create_trace(
                original_count=len(memories_or_messages),
                final_count=0,
                details={"message": "No potential LPCs found"},
            )
            return CompressedMemory(text="", metadata={}), trace

        # Step 2: Score LPCs for Relevance
        scored_lpcs = []
        for lpc in potential_lpcs:
            score = self._score_lpc_relevance(
                lpc, context_query_text, context_query_embedding
            )
            if score >= self.relevance_score_threshold:
                try:
                    tokens = self.tokenizer.count_tokens(lpc.text)
                    scored_lpcs.append(
                        {"lpc": lpc, "score": score, "tokens": tokens}
                    )
                except Exception as e:
                    self.logger.error(f"Failed to count tokens for LPC text: {lpc.text[:100]}... Error: {e}")


        # Step 3: Sort Scored LPCs
        scored_lpcs.sort(key=lambda x: x["score"], reverse=True)

        # Step 4: Select LPCs within Budget
        selected_lpcs_texts: List[str] = []
        selected_lpcs_for_trace: List[Dict[str, Any]] = []
        current_token_count = 0

        for lpc_data in scored_lpcs:
            if len(selected_lpcs_texts) >= self.max_lpcs_to_select:
                self.logger.info(f"Reached max_lpcs_to_select limit: {self.max_lpcs_to_select}")
                break

            lpc_text = lpc_data["lpc"].text
            lpc_tokens = lpc_data["tokens"]

            # Check budget for the LPC itself
            if lpc_tokens > budget: # If an LPC itself is over budget, skip it.
                self.logger.debug(f"LPC skipped (over budget): tokens {lpc_tokens} > budget {budget}")
                continue

            # Check budget for adding this LPC (plus separator if not the first)
            separator_tokens = 0
            if selected_lpcs_texts: # if there's already an LPC, account for separator
                separator_tokens = self.tokenizer.count_tokens(self.lpc_join_separator)

            if current_token_count + lpc_tokens + separator_tokens <= budget:
                selected_lpcs_texts.append(lpc_text)
                current_token_count += lpc_tokens + separator_tokens
                selected_lpcs_for_trace.append(
                    {
                        "id": getattr(lpc_data["lpc"], "memory_id", "unknown_id"),
                        "text_preview": lpc_text[:100] + "...",
                        "score": lpc_data["score"],
                        "tokens": lpc_tokens,
                    }
                )
            else:
                self.logger.info(f"Budget limit reached. Cannot add LPC with {lpc_tokens} tokens. Current tokens: {current_token_count}, Budget: {budget}")
                break

        # Adjust token count if separators were added when they shouldn't have been (only one LPC selected)
        if len(selected_lpcs_texts) == 1 and selected_lpcs_texts:
            current_token_count = self.tokenizer.count_tokens(selected_lpcs_texts[0])


        # Assemble Final Text
        final_text = self.lpc_join_separator.join(selected_lpcs_texts)

        # Generate Trace Information
        trace_details = {
            "strategy_name": self.id,
            "parameters": self._parameters,
            "context_query_preview": context_query_text[:200] + "..."
            if context_query_text
            else "N/A",
            "num_potential_lpcs": len(potential_lpcs),
            "num_scored_above_threshold": len(scored_lpcs),
            "num_selected_lpcs": len(selected_lpcs_texts),
            "total_tokens_selected": current_token_count,
            "budget": budget,
            "selected_lpcs_details": selected_lpcs_for_trace,
        }
        trace = self._create_trace(
            original_count=len(memories_or_messages), # This is total input items
            final_count=len(selected_lpcs_texts),    # This is number of LPCs selected
            details=trace_details,
        )

        self.logger.info(
            f"Selected {len(selected_lpcs_texts)} LPCs with a total of {current_token_count} tokens."
        )

        compressed_memory_metadata = {
            "summary_level": len(selected_lpcs_texts),
            "total_tokens": current_token_count,
            # Potentially add IDs of selected LPCs if useful
        }
        return CompressedMemory(text=final_text, metadata=compressed_memory_metadata), trace # type: ignore

    def decompress(self, compressed_memory: CompressedMemory) -> RawMemory:
        pass


class LPCSelector:
    def __init__(self, strategies: List[LPCSelectorStrategy]):
        self.strategies = strategies

    def select_strategy(self, situation: Any) -> LPCSelectorStrategy:
        # For now, just return the first strategy.
        if not self.strategies:
            raise ValueError("No strategies available to select from.")
        return self.strategies[0]
