from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union, Dict


Number = Union[int, float]


@dataclass
class PromptBudget:
    """Token budget configuration for prompt assembly."""

    query: Optional[Number] = None
    recent_history: Optional[Number] = None
    older_history: Optional[Number] = None
    ltm_snippets: Optional[Number] = None

    def resolve(self, total_tokens: int) -> Dict[str, int]:
        """Return absolute token counts for ``total_tokens``."""

        def _resolve(value: Optional[Number]) -> int:
            if value is None:
                return 0
            if isinstance(value, float):
                return int(total_tokens * value)
            return int(value)

        return {
            "query": _resolve(self.query),
            "recent_history": _resolve(self.recent_history),
            "older_history": _resolve(self.older_history),
            "ltm_snippets": _resolve(self.ltm_snippets),
        }


__all__ = ["PromptBudget"]
