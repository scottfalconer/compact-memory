from __future__ import annotations

"""Abstract interface for memory compression strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Union, Any, Optional


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
    ) -> CompressedMemory:
        """Return compressed form of ``text_or_chunks`` within ``llm_token_budget``."""


__all__ = ["CompressedMemory", "CompressionStrategy"]
