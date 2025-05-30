from __future__ import annotations

"""Render inline memory cues for prototypes."""

import re
from typing import Iterable


class MemoryCueRenderer:
    """Convert prototype summaries into short ``<MEM id=... />`` tags."""

    def __init__(self, max_words: int = 2) -> None:
        self.max_words = max_words

    def _slug(self, text: str) -> str:
        words = re.findall(r"[A-Za-z0-9]+", text)[: self.max_words]
        if not words:
            words = ["mem"]
        return "_".join(words)

    def tag(self, summary: str) -> str:
        return f"<MEM id={self._slug(summary)} />"

    def render(self, summaries: Iterable[str]) -> str:
        return " ".join(self.tag(s) for s in summaries)


__all__ = ["MemoryCueRenderer"]

