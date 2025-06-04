from __future__ import annotations

"""Compression strategy utilities for conversation history."""

from typing import List, Dict, Type, Any, Optional
from compact_memory.chunking import ChunkFn # Added import

from .token_utils import truncate_text
from .compression.strategies_abc import (
    CompressedMemory,
    CompressionStrategy,
    CompressionTrace,
)

try:
    import tiktoken
    _DEFAULT_TOKENIZER = tiktoken.get_encoding("gpt2")
except Exception:  # pragma: no cover - optional dependency
    _DEFAULT_TOKENIZER = None


_COMPRESSION_REGISTRY: Dict[str, Type["CompressionStrategy"]] = {}
_COMPRESSION_INFO: Dict[str, Dict[str, Opt[str]]] = {}


def register_compression_strategy(
    id: str,
    cls: Type["CompressionStrategy"],
    *,
    display_name: str | None = None,
    version: str | None = None,
    source: str = "built-in",
) -> None:
    """Register ``cls`` under ``id`` with optional metadata."""
    prev = _COMPRESSION_INFO.get(id)
    overrides = prev["source"] if prev else None
    _COMPRESSION_REGISTRY[id] = cls
    _COMPRESSION_INFO[id] = {
        "display_name": display_name or id,
        "version": version or "N/A",
        "source": source,
        "overrides": overrides,
    }


def get_compression_strategy(id: str) -> Type["CompressionStrategy"]:
    return _COMPRESSION_REGISTRY[id]


def available_strategies() -> List[str]:
    return sorted(_COMPRESSION_REGISTRY)


def get_strategy_metadata(id: str) -> Dict[str, Opt[str]] | None:
    return _COMPRESSION_INFO.get(id)


def all_strategy_metadata() -> Dict[str, Dict[str, Opt[str]]]:
    return dict(_COMPRESSION_INFO)


# Maintain alias for backward compatibility
LegacyCompressionStrategy = CompressionStrategy


class NoCompression(CompressionStrategy):
    """Simple strategy that joins turns and truncates."""

    id = "none"

    def compress(
        self,
        text: str, # Changed from text_or_chunks
        llm_token_budget: int, # Made non-optional
        chunk_fn: Optional[ChunkFn] = None, # Added chunk_fn
        *,
        tokenizer: Any = None,
        **kwargs: Any,
    ) -> tuple[CompressedMemory, CompressionTrace]:
        tokenizer = tokenizer or _DEFAULT_TOKENIZER or (lambda t, **k: t.split())

        if chunk_fn:
            chunks = chunk_fn(text)
        else:
            chunks = [text]

        processed_text = "\n".join(chunks)

        if llm_token_budget is not None: # Keeping this check as truncate_text might still want Optional
            processed_text = truncate_text(tokenizer, processed_text, llm_token_budget)
        compressed = CompressedMemory(text=processed_text)
        trace = CompressionTrace(
            strategy_name=self.id,
            strategy_params={"llm_token_budget": llm_token_budget},
            input_summary={"input_length": len(processed_text)}, # Use processed_text
            output_summary={"output_length": len(processed_text)}, # Use processed_text
        )
        return compressed, trace


class ImportanceCompression(CompressionStrategy):
    """Apply :func:`dynamic_importance_filter` then truncate."""

    id = "importance"

    def compress(
        self,
        text: str, # Changed from text_or_chunks
        llm_token_budget: int, # Made non-optional
        chunk_fn: Optional[ChunkFn] = None, # Added chunk_fn
        *,
        tokenizer: Any = None,
        **kwargs: Any,
    ) -> tuple[CompressedMemory, CompressionTrace]:
        from .importance_filter import dynamic_importance_filter

        tokenizer = tokenizer or _DEFAULT_TOKENIZER or (lambda t, **k: t.split())

        if chunk_fn:
            chunks = chunk_fn(text)
        else:
            chunks = [text]

        processed_text = dynamic_importance_filter("\n".join(chunks))

        if llm_token_budget is not None: # Keeping this check
            processed_text = truncate_text(tokenizer, processed_text, llm_token_budget)
        compressed = CompressedMemory(text=processed_text)
        trace = CompressionTrace(
            strategy_name=self.id,
            strategy_params={"llm_token_budget": llm_token_budget},
            input_summary={"input_length": len(processed_text)}, # Use processed_text
            output_summary={"output_length": len(processed_text)}, # Use processed_text
        )
        return compressed, trace


register_compression_strategy(NoCompression.id, NoCompression)
register_compression_strategy(ImportanceCompression.id, ImportanceCompression)

__all__ = [
    "CompressionStrategy",
    "NoCompression",
    "ImportanceCompression",
    "register_compression_strategy",
    "get_compression_strategy",
    "available_strategies",
    "get_strategy_metadata",
    "all_strategy_metadata",
]
