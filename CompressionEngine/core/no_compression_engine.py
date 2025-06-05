from __future__ import annotations

"""Simple no-op compression engine."""

from typing import Any, List

from compact_memory.token_utils import truncate_text
from .engines_abc import CompressedMemory, CompressionEngine, CompressionTrace

try:  # pragma: no cover - optional dependency
    import tiktoken

    _DEFAULT_TOKENIZER = tiktoken.get_encoding("gpt2")
except Exception:  # pragma: no cover - optional dependency
    _DEFAULT_TOKENIZER = None


class NoCompressionEngine(CompressionEngine):
    """Engine that returns the text mostly unchanged."""

    id = "no_compression_engine"

    def compress(
        self,
        text_or_chunks: List[str] | str,
        llm_token_budget: int | None,
        *,
        tokenizer: Any = None,
        **kwargs: Any,
    ) -> tuple[CompressedMemory, CompressionTrace]:
        tokenizer = tokenizer or _DEFAULT_TOKENIZER or (lambda t, **_: t.split())
        if isinstance(text_or_chunks, list):
            text = "\n".join(text_or_chunks)
        else:
            text = text_or_chunks
        if llm_token_budget is not None:
            text = truncate_text(tokenizer, text, llm_token_budget)
        compressed = CompressedMemory(text=text)
        trace = CompressionTrace(
            engine_name=self.id,
            engine_params={"llm_token_budget": llm_token_budget},
            input_summary={"input_length": len(text)},
            output_summary={"output_length": len(text)},
        )
        return compressed, trace


__all__ = ["NoCompressionEngine"]
