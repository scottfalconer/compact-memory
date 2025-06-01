from __future__ import annotations

"""Compression strategy utilities for conversation history."""

from typing import List, Optional, Dict, Type, Any

from .token_utils import truncate_text

try:
    import tiktoken
    _DEFAULT_TOKENIZER = tiktoken.get_encoding("gpt2")
except Exception:  # pragma: no cover - optional dependency
    _DEFAULT_TOKENIZER = None


_COMPRESSION_REGISTRY: Dict[str, Type["CompressionStrategy"]] = {}


def register_compression_strategy(id: str, cls: Type["CompressionStrategy"]) -> None:
    """Register ``cls`` under ``id`` for CLI lookup."""
    _COMPRESSION_REGISTRY[id] = cls


def get_compression_strategy(id: str) -> Type["CompressionStrategy"]:
    return _COMPRESSION_REGISTRY[id]


def available_strategies() -> List[str]:
    return sorted(_COMPRESSION_REGISTRY)


class CompressionStrategy:
    """Interface for reducing text to fit a token budget."""

    id: str = "base"

    def compress(
        self,
        texts: List[str],
        tokenizer: Any = None,
        max_tokens: Optional[int] = None,
    ) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class NoCompression(CompressionStrategy):
    """Simple strategy that joins turns and truncates."""

    id = "none"

    def compress(
        self,
        texts: List[str],
        tokenizer: Any = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        tokenizer = tokenizer or _DEFAULT_TOKENIZER or (lambda t, **k: t.split())
        text = "\n".join(texts)
        if max_tokens is not None:
            text = truncate_text(tokenizer, text, max_tokens)
        return text


class ImportanceCompression(CompressionStrategy):
    """Apply :func:`dynamic_importance_filter` then truncate."""

    id = "importance"

    def compress(
        self,
        texts: List[str],
        tokenizer: Any = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        from .importance_filter import dynamic_importance_filter

        tokenizer = tokenizer or _DEFAULT_TOKENIZER or (lambda t, **k: t.split())
        text = dynamic_importance_filter("\n".join(texts))
        if max_tokens is not None:
            text = truncate_text(tokenizer, text, max_tokens)
        return text


register_compression_strategy(NoCompression.id, NoCompression)
register_compression_strategy(ImportanceCompression.id, ImportanceCompression)

__all__ = [
    "CompressionStrategy",
    "NoCompression",
    "ImportanceCompression",
    "register_compression_strategy",
    "get_compression_strategy",
    "available_strategies",
]
