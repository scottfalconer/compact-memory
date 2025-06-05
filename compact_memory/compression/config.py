from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from .strategies_abc import CompressionStrategy


@dataclass
class StrategyConfig:
    """Configuration for creating a :class:`CompressionStrategy`."""

    strategy_name: str
    strategy_params: Dict[str, Any] = field(default_factory=dict)

    def create(self) -> CompressionStrategy:
        """Return a CompressionStrategy object created from this configuration."""
        from . import get_compression_strategy

        cls = get_compression_strategy(self.strategy_name)
        return cls(**self.strategy_params)


__all__ = ["StrategyConfig"]
