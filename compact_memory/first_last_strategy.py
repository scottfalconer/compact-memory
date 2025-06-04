from __future__ import annotations
from typing import List, Union, Any, Optional # Added Optional
from compact_memory.chunking import ChunkFn # Added ChunkFn

from .compression import register_compression_strategy
from .compression.strategies_abc import CompressionStrategy, CompressedMemory, CompressionTrace

class FirstLastStrategy(CompressionStrategy):
    """Keep first and last parts of the text (character-based) within the budget."""

    id = "first_last"

    def compress(
        self,
        text: str, # Changed
        llm_token_budget: int, # Changed (no longer optional)
        chunk_fn: Optional[ChunkFn] = None, # Added
        **kwargs: Any,
    ) -> tuple[CompressedMemory, CompressionTrace]:

        if chunk_fn:
            chunks = chunk_fn(text)
        else:
            chunks = [text]

        processed_text = " ".join(chunks)
        input_len = len(processed_text)

        if llm_token_budget <= 0: # Assuming llm_token_budget is now always int
            kept_text = processed_text
            # For trace, if budget is <=0, it means keep all. "half" isn't well-defined.
            # Let's say trace_half_kept_first is full length, trace_half_kept_last is 0.
            trace_half_intended = input_len
            trace_half_kept_first = input_len
            trace_half_kept_last = 0
        elif input_len <= llm_token_budget:
            kept_text = processed_text
            trace_half_intended = llm_token_budget // 2
            trace_half_kept_first = input_len
            trace_half_kept_last = 0 # Not really a "last half" if whole text is kept
        else:
            # llm_token_budget > 0 and input_len > llm_token_budget
            trace_half_intended = max(llm_token_budget // 2, 0)
            kept_text = processed_text[:trace_half_intended] + processed_text[-trace_half_intended:]
            trace_half_kept_first = len(processed_text[:trace_half_intended])
            trace_half_kept_last = len(processed_text[-trace_half_intended:])
            # Ensure we don't misrepresent if kept_text is shorter due to original text being very short
            # This case is covered by input_len <= llm_token_budget

        compressed = CompressedMemory(text=kept_text)
        trace = CompressionTrace(
            strategy_name=self.id,
            strategy_params={"llm_token_budget": llm_token_budget, "chunked_input": chunk_fn is not None},
            input_summary={"input_length": input_len, "num_chunks": len(chunks)},
            steps=[{
                "type": "first_last_char_slice",
                "intended_half_len": trace_half_intended,
                "kept_first_chars": trace_half_kept_first,
                "kept_last_chars": trace_half_kept_last
            }],
            output_summary={"final_length": len(kept_text)},
            final_compressed_object_preview=kept_text[:50],
        )
        return compressed, trace


register_compression_strategy(FirstLastStrategy.id, FirstLastStrategy)

__all__ = ["FirstLastStrategy"]
