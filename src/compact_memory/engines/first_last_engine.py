from __future__ import annotations

"""Engine that keeps the first and last chunks of text."""

from typing import List, Union, Any, Optional

from compact_memory.token_utils import tokenize_text
import time

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
        """Compress text while keeping the first and last tokens.

        The ``tokenizer`` argument is used only for decoding the kept tokens
        back into text. Tokenization uses ``tiktoken`` if available, otherwise a
        simple whitespace split.
        """

        text = (
            text_or_chunks
            if isinstance(text_or_chunks, str)
            else " ".join(text_or_chunks)
        )

        decode_tokenizer = tokenizer or _DEFAULT_TOKENIZER
        if tokenizer and (
            hasattr(tokenizer, "encode") or hasattr(tokenizer, "tokenize")
        ):
            tokenize_fn = tokenizer
        else:
            tokenize_fn = _DEFAULT_TOKENIZER or (lambda t: t.split())
        tokens = tokenize_text(tokenize_fn, text)

        start = time.monotonic()

        if llm_token_budget is None:
            kept_tokens = tokens
        elif llm_token_budget <= 0:
            kept_tokens = []
        else:
            half = max(llm_token_budget // 2, 0)
            kept_tokens = tokens[:half] + tokens[-half:]

        if hasattr(decode_tokenizer, "decode"):
            try:
                kept = decode_tokenizer.decode(kept_tokens, skip_special_tokens=True)
            except Exception:
                kept = decode_tokenizer.decode(kept_tokens)
        else:
            kept = " ".join(str(t) for t in kept_tokens)

        compressed = CompressedMemory(text=kept)
        trace = CompressionTrace(
            engine_name=self.id,
            strategy_params={"llm_token_budget": llm_token_budget},
            input_summary={"input_length": len(text), "input_tokens": len(tokens)},
            output_summary={
                "final_length": len(kept),
                "final_tokens": len(kept_tokens),
            },
            final_compressed_object_preview=kept[:50],
        )
        trace.add_step(
            "first_last",
            {
                "kept_first_tokens": len(kept_tokens[: len(kept_tokens) // 2]),
                "kept_last_tokens": len(kept_tokens[len(kept_tokens) // 2 :]),
            },
        )
        trace.processing_ms = (time.monotonic() - start) * 1000
        compressed.trace = trace
        compressed.engine_id = self.id
        compressed.engine_config = self.config
        return compressed


# Register engine on import

register_compression_engine(
    FirstLastEngine.id,
    FirstLastEngine,
    display_name="First/Last",
    source="built-in",
)

__all__ = ["FirstLastEngine"]
