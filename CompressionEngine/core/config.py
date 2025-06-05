from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from .engines_abc import CompressionEngine


@dataclass
class EngineConfig:
    """Configuration for creating a :class:`CompressionEngine`."""

    engine_name: str
    engine_params: Dict[str, Any] = field(default_factory=dict)

    def create(self) -> CompressionEngine:
        """Return a CompressionEngine object created from this configuration."""
        from . import get_compression_engine

        cls = get_compression_engine(self.engine_name)
        return cls(**self.engine_params)

@dataclass
class BaseEngineConfig:
    """Base configuration for a CompressionEngine instance."""
    # Add common engine config parameters here if any in the future

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict # Correct import
        return asdict(self) # Correct serialization

__all__ = ["EngineConfig", "BaseEngineConfig"]
