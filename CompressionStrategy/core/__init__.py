"""Compression strategy interfaces and implementations."""

from __future__ import annotations

from .strategies_abc import CompressedMemory, CompressionStrategy, CompressionTrace
from .config import StrategyConfig
from .pipeline_strategy import PipelineCompressionStrategy, PipelineStrategyConfig
from .no_compression import NoCompression
from .registry import (
    register_compression_strategy,
    get_compression_strategy,
    available_strategies,
    get_strategy_metadata,
    all_strategy_metadata,
)

register_compression_strategy(NoCompression.id, NoCompression)
register_compression_strategy(
    PipelineCompressionStrategy.id, PipelineCompressionStrategy
)

__all__ = [
    "CompressedMemory",
    "CompressionStrategy",
    "CompressionTrace",
    "NoCompression",
    "PipelineCompressionStrategy",
    "PipelineStrategyConfig",
    "register_compression_strategy",
    "get_compression_strategy",
    "available_strategies",
    "get_strategy_metadata",
    "all_strategy_metadata",
    "StrategyConfig",
]
