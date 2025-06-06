from __future__ import annotations

from typing import List, Optional, Any, Dict, Union, Tuple

import numpy as np

# Import base classes directly to avoid package import side effects
from .base import BaseCompressionEngine, CompressedMemory, CompressionTrace
from ..active_memory_manager import ActiveMemoryManager, ConversationTurn
from compact_memory.prompt_budget import PromptBudget
from compact_memory.embedding_pipeline import embed_text
from .registry import register_compression_engine

# from compact_memory.token_utils import token_count # token_count is used within ActiveMemoryManager


class ActiveMemoryEngine(BaseCompressionEngine):
    """
    A compression strategy that utilizes an ActiveMemoryManager instance
    to dynamically select and compress conversational context.
    """

    id = "active_memory_neuro"

    def __init__(
        self,
        config_max_history_buffer_turns: int = 100,
        config_prompt_num_forced_recent_turns: int = 0,
        config_prompt_max_activated_older_turns: int = 5,
        config_prompt_activation_threshold_for_inclusion: float = 0.0,
        config_pruning_weight_trace_strength: float = 1.0,
        config_pruning_weight_current_activation: float = 1.0,
        config_pruning_weight_recency: float = 0.1,
        config_initial_activation: float = 0.0,
        config_activation_decay_rate: float = 0.1,
        config_min_activation_floor: float = 0.0,
        config_relevance_boost_factor: float = 1.0,
        prompt_budget: Optional[PromptBudget] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.manager = ActiveMemoryManager(
            config_max_history_buffer_turns=config_max_history_buffer_turns,
            config_prompt_num_forced_recent_turns=config_prompt_num_forced_recent_turns,
            config_prompt_max_activated_older_turns=config_prompt_max_activated_older_turns,
            config_prompt_activation_threshold_for_inclusion=config_prompt_activation_threshold_for_inclusion,
            config_pruning_weight_trace_strength=config_pruning_weight_trace_strength,
            config_pruning_weight_current_activation=config_pruning_weight_current_activation,
            config_pruning_weight_recency=config_pruning_weight_recency,
            config_initial_activation=config_initial_activation,
            config_activation_decay_rate=config_activation_decay_rate,
            config_min_activation_floor=config_min_activation_floor,
            config_relevance_boost_factor=config_relevance_boost_factor,
            prompt_budget=prompt_budget,
        )
        self._strategy_kwargs = {k: v for k, v in kwargs.items()}

    def add_turn(
        self,
        text: str,
        trace_strength: float = 1.0,
        current_activation_level: float = 0.0,  # Will be set by AMM if 0.0
        turn_embedding: Optional[List[float]] = None,
        # **kwargs: Any # Not used for now, but could hold other metadata
    ) -> None:
        """
        Adds a conversational turn to the internal ActiveMemoryManager.
        Embeddings for turns should ideally be generated before calling this,
        or this method could be extended to generate them if not provided.
        """
        # If an embedding is not provided for the turn text, it could be generated here:
        # if turn_embedding is None and text:
        #     turn_embedding_list = embed_text([text])
        #     if turn_embedding_list:
        #         turn_embedding = turn_embedding_list[0]

        new_turn = ConversationTurn(
            text=text,
            trace_strength=trace_strength,
            # If current_activation_level is 0.0, AMM's add_turn will use config_initial_activation
            current_activation_level=(
                current_activation_level
                if current_activation_level != 0.0
                else self.manager.config_initial_activation
            ),
            turn_embedding=turn_embedding,
        )
        self.manager.add_turn(new_turn)

    def compress(
        self,
        text_or_chunks: Union[str, List[str]],
        llm_token_budget: int,
        *,
        tokenizer: Any,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,  # include_meta, trace_events from ABC potentially in kwargs
    ) -> Tuple[CompressedMemory, CompressionTrace]:
        trace_steps = []

        current_query_text = text_or_chunks
        if isinstance(current_query_text, list):
            current_query_text = " ".join(current_query_text)

        trace_steps.append(
            f"Input query for compression context: '{current_query_text}'"
        )
        trace_steps.append(
            f"History length before selection: {len(self.manager.history)}"
        )

        # 1. Embed the current query for relevance boosting
        # Assuming embed_text returns a list of embeddings
        query_embedding_list = (
            embed_text([current_query_text]) if current_query_text else None
        )

        current_query_embedding: Optional[np.ndarray] = None
        if (
            query_embedding_list is not None
            and len(query_embedding_list) > 0
            and query_embedding_list[0] is not None
        ):
            current_query_embedding = np.array(
                query_embedding_list[0], dtype=np.float32
            )
            trace_steps.append("Current query embedded for relevance boosting.")
        else:
            trace_steps.append(
                "Current query could not be embedded or was empty; relevance boosting might be affected."
            )
            # Pass None or an empty array if embedding fails or text is empty.
            # ActiveMemoryManager handles None query embeddings gracefully.

        # 2. Select candidate turns using ActiveMemoryManager
        candidate_turns = self.manager.select_history_candidates_for_prompt(
            current_query_embedding
        )
        trace_steps.append(
            f"Candidate turns selected by ActiveMemoryManager: {len(candidate_turns)}"
        )
        # For more detail: trace_steps.append(f"Candidate turn texts: {[t.text for t in candidate_turns]}")

        # 3. Finalize turns for prompt budget
        final_turns = self.manager.finalize_history_for_prompt(
            candidate_turns,
            max_tokens_for_history=llm_token_budget,
            llm_tokenizer=tokenizer,
        )
        trace_steps.append(
            f"Final turns fitting token budget ({llm_token_budget} tokens): {len(final_turns)}"
        )
        # For more detail: trace_steps.append(f"Final turn texts: {[t.text for t in final_turns]}")

        # 4. Format selected turns into a single string
        compressed_text = "\n".join([turn.text for turn in final_turns])

        # 5. Create CompressedMemory and CompressionTrace
        metadata = {
            "status": "ok",
            "history_length_before_selection": len(self.manager.history),
            "candidate_turns_selected": len(candidate_turns),
            "final_turns_for_prompt": len(final_turns),
            "input_query": current_query_text,
        }
        # if kwargs.get('include_meta', False): # Check how include_meta is passed from caller
        #     metadata["full_trace_steps"] = trace_steps

        compressed_memory = CompressedMemory(text=compressed_text, metadata=metadata)

        # trace_events = kwargs.get('trace_events', []) # Check how trace_events is passed
        # trace_events.extend(trace_steps)
        compression_trace = CompressionTrace(
            engine_name=self.id,
            strategy_params={"llm_token_budget": llm_token_budget},
            input_summary={"history_len": len(self.manager.history)},
            steps=trace_steps,
            output_summary={"compressed_length": len(compressed_text)},
            final_compressed_object_preview=compressed_text[:50],
        )

        return compressed_memory, compression_trace

    def save_learnable_components(self, path: str) -> None:
        """ActiveMemoryManager is rule-based, so no learnable components to save."""
        pass

    def load_learnable_components(self, path: str) -> None:
        """ActiveMemoryManager is rule-based, so no learnable components to load."""
        pass


register_compression_engine(ActiveMemoryEngine.id, ActiveMemoryEngine, source="contrib")

__all__ = ["ActiveMemoryEngine", "ActiveMemoryManager", "ConversationTurn"]
