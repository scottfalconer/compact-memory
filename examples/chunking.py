"""Example chunking functions for Compact Memory."""

from __future__ import annotations

from typing import Callable, List

import tiktoken

ChunkFn = Callable[[str], List[str]]


def newline_splitter(text: str) -> List[str]:
    """Split ``text`` on newlines."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def token_splitter(max_tokens: int = 256) -> ChunkFn:
    """Return a splitter that groups ``max_tokens`` using ``tiktoken``."""
    enc = tiktoken.get_encoding("gpt2")

    def _split(text: str) -> List[str]:
        tokens = enc.encode(text)
        return [
            enc.decode(tokens[i : i + max_tokens])  # noqa: E203
            for i in range(0, len(tokens), max_tokens)
        ]

    return _split


try:  # optional integration
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    def langchain_splitter(chunk_size: int = 4000, chunk_overlap: int = 200) -> ChunkFn:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return lambda text: splitter.split_text(text)

except Exception:  # pragma: no cover - optional
    pass

__all__ = ["ChunkFn", "newline_splitter", "token_splitter", "langchain_splitter"]
