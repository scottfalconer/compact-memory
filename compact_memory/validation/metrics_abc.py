from __future__ import annotations

"""Abstract base class for validation metrics."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imports for type hints only
    from ..compression.strategies_abc import CompressedMemory
    from ..compression.trace import CompressionTrace
else:  # pragma: no cover - avoid runtime dependency
    CompressedMemory = Any
    CompressionTrace = Any


class ValidationMetric(ABC):
    """Interface for evaluating compression strategies and LLM responses."""

    metric_id: str

    def __init__(self, **kwargs: Any) -> None:  # pragma: no cover - simple init
        """Store metric-specific configuration parameters."""
        self.config_params = kwargs

    @abstractmethod
    def evaluate(
        self,
        llm_response: str,
        reference_answer: str,
        original_query: Optional[str] = None,
        compressed_context: Optional[CompressedMemory] = None,
        compression_trace: Optional[CompressionTrace] = None,
        llm_provider_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Return metric results as a dictionary."""
        
