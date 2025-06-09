from __future__ import annotations

"""Engine that keeps the first and last chunks of text."""

from typing import List, Union, Any, Optional

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
        previous_compression_result: Optional[CompressedMemory] = None,
        **kwargs: Any,
    ) -> CompressedMemory:
        text = (
            text_or_chunks
            if isinstance(text_or_chunks, str)
            else " ".join(text_or_chunks)
        )

        tokenizer_for_encoding = _DEFAULT_TOKENIZER or (lambda t: t.split())
        tokens = tokenize_text(tokenizer_for_encoding, text)

        if llm_token_budget is None or llm_token_budget <= 0:
            kept_tokens = tokens
        else:
            half = max(llm_token_budget // 2, 0)
            kept_tokens = tokens[:half] + tokens[-half:]

        # Determine the decoder:
        # The 'tokenizer' kwarg, if provided, is preferred for decoding.
        custom_decoder = tokenizer # 'tokenizer' here is the kwarg passed to compress

        if hasattr(custom_decoder, "decode"):
            try:
                # Attempt to use decode with skip_special_tokens
                kept = custom_decoder.decode(kept_tokens, skip_special_tokens=True)
            except TypeError:
                # If skip_special_tokens is not supported or other TypeError
                try:
                    kept = custom_decoder.decode(kept_tokens)
                except Exception:
                    # Fallback if custom_decoder.decode fails
                    kept = " ".join(str(t) for t in kept_tokens)
            except Exception:
                # Fallback for other exceptions during custom_decoder.decode attempt
                kept = " ".join(str(t) for t in kept_tokens)
        elif callable(custom_decoder):
            # If custom_decoder is a callable (e.g., a lambda or function) but not a full tokenizer object
            try:
                kept = custom_decoder(kept_tokens)
            except Exception:
                # Fallback if calling custom_decoder fails
                kept = " ".join(str(t) for t in kept_tokens)
        elif _DEFAULT_TOKENIZER and hasattr(_DEFAULT_TOKENIZER, "decode"):
            # If no suitable custom_decoder, try the default tokenizer's decode method
            try:
                kept = _DEFAULT_TOKENIZER.decode(kept_tokens)
            except Exception:
                # Fallback if _DEFAULT_TOKENIZER.decode fails
                kept = " ".join(str(t) for t in kept_tokens)
        else:
            # Absolute fallback if no other decoding method is available
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
        compressed.trace = trace
        compressed.engine_id = self.id
        compressed.engine_config = self.config
        return compressed


# register_compression_engine(FirstLastEngine.id, FirstLastEngine, source="contrib") # Removed, as it's now registered in engines/__init__.py

__all__ = ["FirstLastEngine"]
