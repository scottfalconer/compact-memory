from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ConversationTurn:
    """Simple container for a conversation turn."""

    text: str
    trace_strength: float = 0.0
    current_activation_level: float = 0.0


@dataclass
class ActiveMemoryManager:
    """Manage a history buffer of conversation turns."""

    config_max_history_buffer_turns: int = 100
    config_prompt_num_forced_recent_turns: int = 0
    config_pruning_weight_trace_strength: float = 1.0
    config_pruning_weight_current_activation: float = 1.0
    config_pruning_weight_recency: float = 0.1
    history: List[ConversationTurn] = field(default_factory=list)

    # --------------------------------------------------------------
    def add_turn(self, turn: ConversationTurn) -> None:
        """Add ``turn`` to the history, pruning if necessary."""
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
                score = (
                    self.config_pruning_weight_trace_strength * t.trace_strength
                    + self.config_pruning_weight_current_activation
                    * t.current_activation_level
                    + self.config_pruning_weight_recency * recency
                )
                scores.append(score)
            min_index = scores.index(min(scores))
            candidates.pop(min_index)

        self.history = candidates + keep_slice


__all__ = ["ConversationTurn", "ActiveMemoryManager"]
