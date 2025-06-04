from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Type

import tiktoken

from .spacy_utils import get_nlp


_CHUNKER_REGISTRY: Dict[str, Type["Chunker"]] = {}


def register_chunker(id: str, cls: Type["Chunker"]) -> None:
    _CHUNKER_REGISTRY[id] = cls


class Chunker(ABC):
    """Interface for chunking text."""

    id: str

    @abstractmethod
    def chunk(self, text: str) -> List[str]: ...  # noqa: E704

    def config(self) -> Dict[str, int | str]:
        return {}


class SentenceWindowChunker(Chunker):
    """Split text into sentence windows with optional overlap."""

    id = "sentence_window"

    def __init__(self, max_tokens: int = 256, overlap_tokens: int = 32):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        try:
            self.tokenizer = tiktoken.get_encoding("gpt2")
        except Exception:  # pragma: no cover - offline fallback
            self.tokenizer = None

    def config(self) -> Dict[str, int | str]:
        return {
            "id": self.id,
            "max_tokens": self.max_tokens,
            "overlap_tokens": self.overlap_tokens,
        }

    def _sentences(self, text: str) -> List[str]:
        doc = get_nlp()(text.strip())
        return [s.text.strip() for s in doc.sents if s.text.strip()]

    def chunk(self, text: str) -> List[str]:
        sents = self._sentences(text)
        if not sents:
            return []
        chunks: List[List[int]] = []
        current: List[int] = []
        for sent in sents:
            if self.tokenizer is not None:
                tokens = self.tokenizer.encode(sent)
            else:
                tokens = sent.split()
            if len(tokens) > self.max_tokens:
                logging.warning("long sentence split")
                for i in range(0, len(tokens), self.max_tokens):
                    sub = tokens[i : i + self.max_tokens]  # noqa: E203
                    if current:
                        chunks.append(current)
                        current = []
                    chunks.append(sub)
                continue
            if len(current) + len(tokens) > self.max_tokens:
                chunks.append(current)
                overlap = current[-self.overlap_tokens :] if self.overlap_tokens else []
                current = overlap + tokens
            else:
                current.extend(tokens)
        if current:
            chunks.append(current)
        if self.tokenizer is not None:
            return [self.tokenizer.decode(c) for c in chunks]
        else:
            return [" ".join(c) for c in chunks]


class FixedSizeChunker(Chunker):
    """Fallback chunker splitting by characters."""

    id = "fixed_size"

    def __init__(self, size: int = 1024):
        self.size = size

    def config(self) -> Dict[str, int | str]:
        return {"id": self.id, "size": self.size}

    def chunk(self, text: str) -> List[str]:
        return [
            text[i : i + self.size] for i in range(0, len(text), self.size)
        ]  # noqa: E203


class AgenticChunker(Chunker):
    """Segment text into belief-sized ideas using :func:`agentic_split`."""

    id = "agentic"

    def __init__(self, max_tokens: int = 120, sim_threshold: float = 0.3) -> None:
        self.max_tokens = max_tokens
        self.sim_threshold = sim_threshold

    def config(self) -> Dict[str, int | str]:
        return {
            "id": self.id,
            "max_tokens": self.max_tokens,
            "sim_threshold": self.sim_threshold,
        }

    def chunk(self, text: str) -> List[str]:
        from .segmentation import agentic_split

        return agentic_split(
            text, max_tokens=self.max_tokens, sim_threshold=self.sim_threshold
        )


register_chunker(SentenceWindowChunker.id, SentenceWindowChunker)
register_chunker(FixedSizeChunker.id, FixedSizeChunker)
register_chunker(AgenticChunker.id, AgenticChunker)

__all__ = [
    "Chunker",
    "SentenceWindowChunker",
    "FixedSizeChunker",
    "AgenticChunker",
    "register_chunker",
    "_CHUNKER_REGISTRY",
]
