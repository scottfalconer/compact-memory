from __future__ import annotations

"""Engine that keeps the first and last chunks of text."""

from typing import List, Union, Any

from compact_memory.token_utils import tokenize_text

try:  # pragma: no cover - optional dependency
    import tiktoken

    _DEFAULT_TOKENIZER = tiktoken.get_encoding("gpt2")
except Exception:  # pragma: no cover - optional dependency
    _DEFAULT_TOKENIZER = None

# Import directly from base to avoid package import side effects
from .base import BaseCompressionEngine, CompressedMemory, CompressionTrace
from .registry import register_compression_engine


class FirstLastEngine(BaseCompressionEngine):
    """Keep first and last parts of the text within the budget."""

    id = "first_last"

    def compress(
        self,
        text_or_chunks: Union[str, List[str]],
        llm_token_budget: int | None,
        *,
        tokenizer: Any = None,
        **kwargs: Any,
    ) -> tuple[CompressedMemory, CompressionTrace]:
        text = (
            text_or_chunks
            if isinstance(text_or_chunks, str)
            else " ".join(text_or_chunks)
        )

        tokenizer = tokenizer or _DEFAULT_TOKENIZER or (lambda t: t.split())
        tokens = tokenize_text(tokenizer, text)

        if llm_token_budget is None or llm_token_budget <= 0:
            kept_tokens = tokens
        else:
            half = max(llm_token_budget // 2, 0)
            kept_tokens = tokens[:half] + tokens[-half:]

        if hasattr(tokenizer, "decode"):
            try:  # Prefer to drop special tokens if supported
                kept = tokenizer.decode(kept_tokens, skip_special_tokens=True)
            except TypeError:  # e.g. tiktoken decode has no skip_special_tokens
                try:
                    kept = tokenizer.decode(kept_tokens)
                except Exception:
                    kept = " ".join(str(t) for t in kept_tokens)
            except Exception:
                kept = " ".join(str(t) for t in kept_tokens)
        else:
            kept = " ".join(str(t) for t in kept_tokens)

        compressed = CompressedMemory(text=kept)
        trace = CompressionTrace(
            engine_name=self.id,
            strategy_params={"llm_token_budget": llm_token_budget},
            input_summary={"input_length": len(text), "input_tokens": len(tokens)},
            steps=[
                {
                    "type": "first_last",
                    "kept_first_tokens": len(kept_tokens[: len(kept_tokens) // 2]),
                    "kept_last_tokens": len(kept_tokens[len(kept_tokens) // 2 :]),
                }
            ],
            output_summary={
                "final_length": len(kept),
                "final_tokens": len(kept_tokens),
            },
            final_compressed_object_preview=kept[:50],
        )
        return compressed, trace


# register_compression_engine(FirstLastEngine.id, FirstLastEngine, source="contrib") # Removed, as it's now registered in engines/__init__.py

__all__ = ["FirstLastEngine"]
