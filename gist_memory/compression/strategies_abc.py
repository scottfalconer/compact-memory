from __future__ import annotations

"""Abstract interface for memory compression strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Union, Any, Optional, Tuple

from .trace import CompressionTrace


@dataclass
class CompressedMemory:
    """Simple container for compressed memory text and metadata."""

    text: str
    metadata: Optional[Dict[str, Any]] = None


class CompressionStrategy(ABC):
    """Interface for memory compression algorithms."""

    id: str

    @abstractmethod
    def compress(
        self,
        text_or_chunks: Union[str, List[str]],
        llm_token_budget: int,
        **kwargs: Any,
    ) -> Tuple[CompressedMemory, CompressionTrace]:
        """Return compressed form of ``text_or_chunks`` and trace.

        The ``CompressedMemory`` holds the data to be given to an LLM while
        ``CompressionTrace`` captures decision steps taken during compression.
        """


__all__ = ["CompressedMemory", "CompressionStrategy", "CompressionTrace"]
