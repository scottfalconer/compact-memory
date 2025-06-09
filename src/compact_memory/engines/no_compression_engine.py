from __future__ import annotations

"""Simple no-op compression engine."""

import time
import logging  # Moved import logging to top
from typing import Any, List, Optional, Union  # Added Union for text_or_chunks

from compact_memory.token_utils import truncate_text


# Import directly from base to avoid triggering package-level side effects
from .base import BaseCompressionEngine, CompressedMemory, CompressionTrace
from .registry import register_compression_engine

try:  # pragma: no cover - optional dependency
    import tiktoken

    _DEFAULT_TOKENIZER: Optional[tiktoken.Encoding] = tiktoken.get_encoding("gpt2")
except Exception:  # pragma: no cover - optional dependency
    _DEFAULT_TOKENIZER = None


class NoCompressionEngine(BaseCompressionEngine):
    """Engine that returns the text mostly unchanged."""

    id = "none"

    def compress(
        self,
        text_or_chunks: Union[str, List[str]],  # Used Union explicitly
        budget: int | None,
        *,
        tokenizer: Any = None,
        previous_compression_result: Optional[CompressedMemory] = None,
        **kwargs: Any,
    ) -> CompressedMemory:
        """
        Returns the input text largely unchanged.

        If a `budget` is provided, the text will be truncated to fit.
        Otherwise, the original text is returned.
        Trace generation is controlled by `self.config.enable_trace`.
        If False, `CompressedMemory.trace` will be None.
        """
        logging.debug(f"NoCompressionEngine: Processing text, budget: {budget}")
        # Ensure tokenizer is not None for truncate_text if budget is not None
        active_tokenizer: Any = tokenizer or _DEFAULT_TOKENIZER
        if (
            active_tokenizer is None and budget is not None
        ):  # If no tokenizer and budget is set, truncation might behave unexpectedly or fail
            active_tokenizer = (
                lambda t, **_: t.split()
            )  # Fallback to simple split for truncate_text's tokenizer requirement

        start_time = time.monotonic()

        input_text_content: str
        if isinstance(text_or_chunks, list):
            input_text_content = "\n".join(text_or_chunks)
        else:
            input_text_content = text_or_chunks

        # 'final_text' will be the text after potential truncation
        final_text = input_text_content
        if budget is not None:
            final_text = truncate_text(active_tokenizer, input_text_content, budget)

        if not self.config.enable_trace:
            return CompressedMemory(
                text=final_text,
                trace=None,
                engine_id=self.id,
                engine_config=self.config.model_dump(mode="json"),
            )

        # Proceed with trace generation
        trace = CompressionTrace(
            engine_name=self.id,
            strategy_params={"llm_token_budget": budget},
            input_summary={"input_length": len(input_text_content)},  # Original length
            output_summary={"output_length": len(final_text)},  # Final length
            final_compressed_object_preview=final_text[:50],
        )
        trace.add_step(
            "truncate_content" if budget is not None else "no_op",
            {
                "llm_token_budget": budget,
                "result_length": len(final_text),
            },
        )
        trace.processing_ms = (time.monotonic() - start_time) * 1000

        return CompressedMemory(
            text=final_text,
            trace=trace,
            engine_id=self.id,
            engine_config=self.config.model_dump(mode="json"),
        )


__all__ = ["NoCompressionEngine"]

# Self-register on import so the engine is available without
# explicit registration elsewhere.
register_compression_engine(
    NoCompressionEngine.id,
    NoCompressionEngine,
    display_name="No Compression",
    source="built-in",
)
