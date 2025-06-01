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
        """Return compressed form of ``text_or_chunks`` and a ``CompressionTrace``.

        The trace should record input and output token/character counts and the
        time spent compressing so efficiency can be measured.
        """

    def save_learnable_components(self, path: str) -> None:  # pragma: no cover - optional
        """Persist any trainable state to ``path``."""

    def load_learnable_components(self, path: str) -> None:  # pragma: no cover - optional
        """Load previously saved trainable state from ``path``."""


__all__ = ["CompressedMemory", "CompressionStrategy", "CompressionTrace"]
