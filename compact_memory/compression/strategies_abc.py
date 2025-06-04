from __future__ import annotations

"""Abstract interface for memory compression strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Union, Any, Optional, Tuple
from compact_memory.chunking import ChunkFn # Added import

from .trace import CompressionTrace


@dataclass
class CompressedMemory:
    """
    Simple container for compressed memory text and associated metadata.

    Attributes:
        text: The compressed string output from a compression strategy.
        metadata: An optional dictionary to hold any additional information
                  about the compressed content, such as source IDs, timestamps,
                  or strategy-specific details.
    """
    text: str
    metadata: Optional[Dict[str, Any]] = None


class CompressionStrategy(ABC):
    """
    Abstract Base Class for defining memory compression strategies.

    Strategy developers should subclass this class and implement the `compress` method.
    Each strategy must also have a unique `id` class attribute, which is a string
    used by the framework to identify and register the strategy.

    Example:
    ```python
    from compact_memory.compression.strategies_abc import CompressionStrategy, CompressedMemory, CompressionTrace

    class MyCustomStrategy(CompressionStrategy):
        id = "my_custom_strategy"

        def compress(self, text_or_chunks, llm_token_budget, **kwargs):
            # ... implementation ...
            compressed_text = "..."
            trace_details = CompressionTrace(...)
            return CompressedMemory(text=compressed_text), trace_details
    ```

    Optionally, strategies can implement `save_learnable_components` and
    `load_learnable_components` if they involve trainable models or state
    that needs to be persisted and reloaded.
    """

    id: str # Unique string identifier for the strategy.

    @abstractmethod
    def compress(
        self,
        text: str, # Changed from text_or_chunks
        llm_token_budget: int,
        chunk_fn: Optional[ChunkFn] = None, # Added chunk_fn
        **kwargs: Any,
    ) -> Tuple[CompressedMemory, CompressionTrace]:
        """
        Compresses the input text to meet the LLM token budget.

        This method must be implemented by concrete strategy subclasses.
        If `chunk_fn` is provided, it's used to split `text` into chunks before compression.
        Otherwise, `text` is treated as a single chunk.

        Args:
            text: The input text to be compressed.
            llm_token_budget: The target maximum number of tokens (or a similar unit
                              like characters, depending on the strategy's internal logic)
                              that the compressed output should ideally have. The strategy
                              should strive to keep the output within this budget.
            chunk_fn: Optional function to split the text into chunks.
                      If None, the text is processed as a single unit.
            **kwargs: Additional keyword arguments that specific strategies might require
                      or that the calling framework might provide. Common examples include:
                      - `tokenizer`: A tokenizer instance (e.g., from Hugging Face) that
                        can be used for accurate token counting and text processing.
                      - `metadata`: Any other relevant metadata that might influence the
                        compression process.

        Returns:
            A tuple containing:
                - CompressedMemory: An object holding the `text` attribute with the
                                  compressed string result, and optionally `metadata`.
                - CompressionTrace: An object logging details about the compression
                                  process (e.g., strategy name, parameters, input/output
                                  token counts, steps taken, processing time). This is crucial
                                  for debugging, analysis, and understanding strategy behavior.
        """

    def save_learnable_components(self, path: str) -> None:  # pragma: no cover - optional
        """Saves any learnable components of the strategy to the specified path."""

    def load_learnable_components(self, path: str) -> None:  # pragma: no cover - optional
        """Loads any learnable components of the strategy from the specified path."""


__all__ = ["CompressedMemory", "CompressionStrategy", "CompressionTrace"]
