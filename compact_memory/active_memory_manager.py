from __future__ import annotations

from dataclasses import dataclass, field # Keep dataclass if ConversationTurn remains here
from typing import List, Optional, Dict # Added Dict

from .prompt_budget import PromptBudget
# Assuming ConversationTurn will come from models.py
from .models import ConversationTurn
from .vector_store import BaseVectorStore, InMemoryVectorStore # Added
from .token_utils import token_count

import numpy as np
import logging
import uuid # For turn IDs if not already present


# ConversationTurn is now imported from models.py, remove local definition.

@dataclass
class ActiveMemoryManager:
    """Manage a history buffer of conversation turns.
    Optionally, can store and query turn embeddings in a vector store.
    """
    vector_store: BaseVectorStore
    embedding_dim: int # Must be provided if vector_store is used for turns

    # Configuration for history management
    config_max_history_buffer_turns: int = 100
    config_prompt_num_forced_recent_turns: int = 0
    config_prompt_max_activated_older_turns: int = 5
    config_prompt_activation_threshold_for_inclusion: float = 0.0
    config_pruning_weight_trace_strength: float = 1.0
    config_pruning_weight_current_activation: float = 1.0
    config_pruning_weight_recency: float = 0.1
    config_initial_activation: float = 0.0
    config_activation_decay_rate: float = 0.1
    config_min_activation_floor: float = 0.0
    config_relevance_boost_factor: float = 1.0

    history: List[ConversationTurn] = field(default_factory=list)
    prompt_budget: Optional[PromptBudget] = None
    store_turn_embeddings: bool = False # Flag to control if AMM uses its vector_store for turns

    def __post_init__(self):
        # This allows vector_store to be None if store_turn_embeddings is False,
        # but BaseVectorStore type hint implies it's always there.
        # For clarity, Agent will always pass a vector_store.
        # If store_turn_embeddings is True, embedding_dim must be valid.
        if self.store_turn_embeddings and self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive if store_turn_embeddings is True.")

    # --------------------------------------------------------------
    def add_turn(self, turn: ConversationTurn) -> None:
        """Add ``turn`` to the history, pruning if necessary.
        If store_turn_embeddings is True, adds the turn's embedding to the vector store.
        """
        self._decay_activations()

        # Ensure turn has an ID. The model in models.py provides a default factory.
        if not turn.turn_id: # Should always have one from Pydantic model
            turn.turn_id = uuid.uuid4().hex

        if turn.current_activation_level == 0.0: # Pydantic model default is 0.0
            turn.current_activation_level = self.config_initial_activation

        self.history.append(turn)

        if self.store_turn_embeddings and turn.turn_embedding:
            try:
                embedding_np = np.array(turn.turn_embedding, dtype=np.float32)
                if embedding_np.ndim == 1:
                     if embedding_np.shape[0] != self.embedding_dim:
                        logging.warning(f"Turn {turn.turn_id} embedding_dim {embedding_np.shape[0]} != AMM embedding_dim {self.embedding_dim}. Skipping add_vector.")
                     else:
                        self.vector_store.add_vector(
                            id=turn.turn_id,
                            vector=embedding_np,
                            metadata={"text": turn.user_message + "\n" + turn.agent_response} # Example metadata
                        )
                else:
                    logging.warning(f"Turn {turn.turn_id} embedding is not 1D. Skipping add_vector.")
            except Exception as e:
                logging.error(f"Error adding turn embedding to vector store: {e}")


        if len(self.history) > self.config_max_history_buffer_turns:
            removed_turn = self._prune_history_buffer()
            if self.store_turn_embeddings and removed_turn and removed_turn.turn_id:
                try:
                    self.vector_store.delete_vector(removed_turn.turn_id)
                except Exception as e:
                    logging.error(f"Error deleting turn embedding from vector store: {e}")

    # --------------------------------------------------------------
    def _prune_history_buffer(self) -> None:
        """Prune the history buffer using weighted retention scores."""
        max_turns = self.config_max_history_buffer_turns
        forced_recent = self.config_prompt_num_forced_recent_turns

        if len(self.history) <= max_turns:
            return

        # Slice off the forced recent turns which are never removed
        if forced_recent > 0:
            keep_slice = self.history[-forced_recent:]
            candidates = self.history[:-forced_recent]
        else:
            keep_slice = []
            candidates = list(self.history)

        while len(candidates) + len(keep_slice) > max_turns and candidates:
            n = len(candidates)
            # Compute retention scores taking recency into account
            scores = []
            for idx, t in enumerate(candidates):
                if n == 1:
                    recency = 1.0
                else:
                    recency = idx / (n - 1)
                w_ts = self.config_pruning_weight_trace_strength
                w_ca = self.config_pruning_weight_current_activation
                w_rec = self.config_pruning_weight_recency
                score = (
                    w_ts * t.trace_strength
                    + w_ca * t.current_activation_level
                    + w_rec * recency
                )
                scores.append(score)
            min_index = scores.index(min(scores))
            removed_candidate = candidates.pop(min_index)
            logging.debug("[prune] removing turn with score %.3f", scores[min_index])
            # Return the removed turn so its vector can be deleted if needed
            return removed_candidate

        self.history = candidates + keep_slice
        return None # No turn was removed if condition not met

    # --------------------------------------------------------------
    def _decay_activations(self) -> None:
        """Decay activation levels using ``config_activation_decay_rate``."""
        rate = self.config_activation_decay_rate
        floor = self.config_min_activation_floor
        for t in self.history:
            before = t.current_activation_level
            t.current_activation_level = max(
                floor, t.current_activation_level * (1.0 - rate)
            )
            logging.debug(
                "[decay] %s %.3f -> %.3f",
                t.text[:20],
                before,
                t.current_activation_level,
            )

    # --------------------------------------------------------------
    def boost_activation_by_relevance(
        self, current_query_embedding: np.ndarray
    ) -> None:
        """Boost activation via similarity to ``current_query_embedding``."""
        if current_query_embedding is None:
            return

        q = np.asarray(current_query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q) or 1.0
        q = q / q_norm

        for t in self.history:
            if t.turn_embedding is None:
                continue
            vec = np.asarray(t.turn_embedding, dtype=np.float32)
            v_norm = np.linalg.norm(vec) or 1.0
            vec = vec / v_norm
            similarity = float(np.dot(vec, q))
            boost = similarity * self.config_relevance_boost_factor
            t.current_activation_level += boost
            if t.current_activation_level < self.config_min_activation_floor:
                t.current_activation_level = self.config_min_activation_floor
            logging.debug(
                "[boost] %s sim=%.3f new_level=%.3f",
                t.text[:20],
                similarity,
                t.current_activation_level,
            )

    # --------------------------------------------------------------
    def select_history_candidates_for_prompt(
        self, current_query_embedding: np.ndarray
    ) -> List[ConversationTurn]:
        """Return recent turns plus older activated ones for STM."""

        self.boost_activation_by_relevance(current_query_embedding)

        num_recent = self.config_prompt_num_forced_recent_turns
        if num_recent > 0:
            recent_slice = self.history[-num_recent:]
            older = self.history[:-num_recent]
        else:
            recent_slice = []
            older = list(self.history)

        threshold = self.config_prompt_activation_threshold_for_inclusion
        activated = []
        for t in older:
            if t.current_activation_level >= threshold:
                activated.append(t)

        activated.sort(
            key=lambda t: (t.current_activation_level, t.trace_strength),
            reverse=True,
        )

        max_older = self.config_prompt_max_activated_older_turns
        selected_older = activated[:max_older]

        # preserve original order of selected older turns
        selected_older.sort(key=lambda t: self.history.index(t))

        selected = selected_older + list(recent_slice)
        logging.debug(
            "[prompt] select_history candidates=%d recent=%d older=%d",
            len(selected),
            len(recent_slice),
            len(selected_older),
        )
        return selected

    # --------------------------------------------------------------
    def finalize_history_for_prompt(
        self,
        candidate_turns: List[ConversationTurn],
        max_tokens_for_history: int,
        llm_tokenizer,
    ) -> List[ConversationTurn]:
        """Fit ``candidate_turns`` within ``max_tokens_for_history`` tokens."""

        forced_recent = self.config_prompt_num_forced_recent_turns
        if forced_recent > 0:
            recent = candidate_turns[-forced_recent:]
            older = candidate_turns[:-forced_recent]
        else:
            recent = []
            older = list(candidate_turns)

        priority = list(recent) + older

        current_tokens = 0
        kept_ids = set()

        for turn in priority:
            if hasattr(turn, "text"):
                text = turn.text
            else:
                user = getattr(turn, "user_message", "")
                agent = getattr(turn, "agent_response", "")
                text = f"{user}\n{agent}".strip()

            n_tokens = token_count(llm_tokenizer, text)

            if current_tokens + n_tokens <= max_tokens_for_history:
                kept_ids.add(id(turn))
                current_tokens += n_tokens
            else:
                continue

        return [t for t in candidate_turns if id(t) in kept_ids]


__all__ = ["ConversationTurn", "ActiveMemoryManager"]
