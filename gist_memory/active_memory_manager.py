from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class ConversationTurn:
    """Simple container for a conversation turn."""

    text: str
    trace_strength: float = 0.0
    current_activation_level: float = 0.0
    turn_embedding: Optional[List[float]] = None


@dataclass
class ActiveMemoryManager:
    """Manage a history buffer of conversation turns."""

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

    # --------------------------------------------------------------
    def add_turn(self, turn: ConversationTurn) -> None:
        """Add ``turn`` to the history, pruning if necessary."""
        self._decay_activations()
        if turn.current_activation_level == 0.0:
            turn.current_activation_level = self.config_initial_activation
        self.history.append(turn)
        if len(self.history) > self.config_max_history_buffer_turns:
            self._prune_history_buffer()

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
            candidates.pop(min_index)

        self.history = candidates + keep_slice

    # --------------------------------------------------------------
    def _decay_activations(self) -> None:
        """Decay activation levels using ``config_activation_decay_rate``."""
        rate = self.config_activation_decay_rate
        floor = self.config_min_activation_floor
        for t in self.history:
            t.current_activation_level = max(
                floor, t.current_activation_level * (1.0 - rate)
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

        return selected_older + list(recent_slice)

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

            if hasattr(llm_tokenizer, "tokenize"):
                try:
                    tokens = llm_tokenizer.tokenize(text)
                except Exception:
                    tokens = llm_tokenizer(text, return_tensors=None).get(
                        "input_ids", []
                    )
            else:
                tokens = llm_tokenizer(text, return_tensors=None).get(
                    "input_ids", []
                )

            if isinstance(tokens, (list, tuple)):
                if tokens and isinstance(tokens[0], (list, tuple)):
                    token_list = list(tokens[0])
                else:
                    token_list = list(tokens)
            else:
                token_list = [tokens]
            n_tokens = len(token_list)

            if current_tokens + n_tokens <= max_tokens_for_history:
                kept_ids.add(id(turn))
                current_tokens += n_tokens
            else:
                break

        return [t for t in candidate_turns if id(t) in kept_ids]


__all__ = ["ConversationTurn", "ActiveMemoryManager"]
