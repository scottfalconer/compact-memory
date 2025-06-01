from __future__ import annotations

"""Abstract base class for validation metrics."""

from abc import ABC, abstractmethod
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from ..memory_creation import CompressedMemoryObject
else:  # pragma: no cover - avoid runtime dependency
    CompressedMemoryObject = Any


class ValidationMetric(ABC):
    """Interface for evaluating compression strategies and LLM responses."""

    @abstractmethod
    def evaluate(
        self,
        original_query: str,
        llm_response: str,
        compressed_context: CompressedMemoryObject,
        reference_data: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Return metric results as a dictionary."""
