from __future__ import annotations

"""Engine that keeps the first and last chunks of text."""

import time
import logging # Moved import logging to top
from typing import List, Union, Any, Optional

from compact_memory.token_utils import tokenize_text


try:  # pragma: no cover - optional dependency
    import tiktoken
    _DEFAULT_TOKENIZER: Optional[tiktoken.Encoding] = tiktoken.get_encoding("gpt2")
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
        budget: Optional[int], # Changed from int | None
        *,
        tokenizer: Any = None,
        previous_compression_result: Optional[CompressedMemory] = None,
        **kwargs: Any,
    ) -> CompressedMemory:
        """Compress text while keeping the first and last tokens.

        The ``tokenizer`` argument is used only for decoding the kept tokens
        back into text. Tokenization uses ``tiktoken`` if available, otherwise a
        simple whitespace split.
        If `self.config.enable_trace` is False, the `trace` field in the
        returned `CompressedMemory` object will be None.
        """
        logging.debug(f"FirstLastEngine: Compressing text, budget: {budget}")

        input_text = ( # Renamed from text to avoid conflict with text kwarg for Pydantic model
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
        tokens = tokenize_text(tokenize_fn, input_text)

        start_time = time.monotonic() # Renamed from start

        if budget is None:
            kept_tokens = tokens
        elif budget <= 0:
            kept_tokens = []
        else:
            half = max(budget // 2, 0)
            kept_tokens = tokens[:half] + tokens[-half:]

        # Ensure decode_tokenizer is not None and has decode method
        if decode_tokenizer is not None and hasattr(decode_tokenizer, "decode"):
            try:
                kept = decode_tokenizer.decode(kept_tokens, skip_special_tokens=True)
            except Exception:
                # Fallback if skip_special_tokens causes issues or other decode error
                kept = decode_tokenizer.decode(kept_tokens)
        else:
            # Fallback if no valid decoder, join tokens which might be strings or ints
            kept = " ".join(map(str, kept_tokens))


        if not self.config.enable_trace:
            return CompressedMemory(
                text=kept,
                trace=None,
                engine_id=self.id,
                engine_config=self.config.model_dump(mode='json')
            )

        # Proceed with trace generation
        trace = CompressionTrace(
            engine_name=self.id,
            strategy_params={"budget": budget}, # Use 'budget'
            input_summary={"input_length": len(input_text), "input_tokens": len(tokens)},
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
        trace.processing_ms = (time.monotonic() - start_time) * 1000 # Use start_time

        return CompressedMemory(
            text=kept,
            trace=trace,
            engine_id=self.id,
            engine_config=self.config.model_dump(mode='json')
        )


# Register engine on import

register_compression_engine(
    FirstLastEngine.id,
    FirstLastEngine,
    display_name="First/Last",
    source="built-in",
)

__all__ = ["FirstLastEngine"]
