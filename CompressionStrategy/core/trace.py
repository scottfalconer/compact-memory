"""Data structures for tracking compression steps and efficiency metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CompressionTrace:
    """
    Represents the chain-of-thought and operational log of a CompressionStrategy.

    This data structure is used to record the inputs, outputs, intermediate steps,
    and performance metrics of a compression process. It is invaluable for
    debugging, analyzing strategy behavior, and understanding the impact of
    different parameters or techniques.

    Attributes:
        strategy_name: The `id` of the `CompressionStrategy` used.
        strategy_params: A dictionary of parameters with which the strategy was
                         configured or called for this specific compression task.
        input_summary: A dictionary summarizing the input to the compression.
                       Common keys include `original_length` (e.g., character count)
                       and `original_tokens` (if a tokenizer was used).
        steps: A list of dictionaries, where each dictionary represents a distinct
               step or operation performed by the strategy during compression.
               Each step entry should ideally have a 'type' field (e.g., "chunking",
               "summarization_llm_call", "filtering") and a 'details' field
               containing relevant information about that step (e.g., number of
               chunks created, prompt sent to LLM, items filtered out).
        output_summary: A dictionary summarizing the output of the compression.
                        Common keys include `compressed_length` (e.g., character count)
                        and `compressed_tokens` (if a tokenizer was used).
        processing_ms: The total time taken for the compression process, in milliseconds.
                       This should be populated by the strategy if possible.
        final_compressed_object_preview: A short string preview of the final
                                         compressed text, useful for quick inspection in logs.
    """

    strategy_name: str
    strategy_params: Dict[str, Any]
    input_summary: Dict[str, Any]
    steps: List[Dict[str, Any]] = field(default_factory=list)
    output_summary: Dict[str, Any] = field(default_factory=dict)
    processing_ms: float | None = None
    final_compressed_object_preview: Optional[str] = None


__all__ = ["CompressionTrace"]
