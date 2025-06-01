"""Data structures for tracking compression steps."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CompressionTrace:
    """Represents the chain-of-thought of a :class:`CompressionStrategy`."""

    strategy_name: str
    strategy_params: Dict[str, Any]
    input_summary: Dict[str, Any]
    steps: List[Dict[str, Any]] = field(default_factory=list)
    output_summary: Dict[str, Any] = field(default_factory=dict)
    final_compressed_object_preview: Optional[str] = None


__all__ = ["CompressionTrace"]
