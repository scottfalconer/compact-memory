from __future__ import annotations

"""Placeholder LLM summarising chunker for future experimentation."""

from typing import List

from compact_memory.chunker import Chunker


class LLMSummarisingChunker(Chunker):
    """Chunker that would summarize text with an LLM."""

    id = "llm_summary"

    def chunk(self, text: str) -> List[str]:  # pragma: no cover - experimental
        raise NotImplementedError
