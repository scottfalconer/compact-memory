"""Compatibility layer for legacy contrib strategies.

These strategies have moved to ``compact_memory.strategies.experimental``.
"""

from __future__ import annotations

from compact_memory.compression import register_compression_strategy


def enable_all_contrib_strategies() -> None:
    """Register all strategies contained in :mod:`contrib`."""
    from .strategies import (
        ChainedStrategy,
        RationaleEpisodeStrategy,
        PrototypeSystemStrategy,
    )

    register_compression_strategy(ChainedStrategy.id, ChainedStrategy, source="contrib")
    register_compression_strategy(
        RationaleEpisodeStrategy.id, RationaleEpisodeStrategy, source="contrib"
    )
    register_compression_strategy(
        PrototypeSystemStrategy.id, PrototypeSystemStrategy, source="contrib"
    )


__all__ = ["enable_all_contrib_strategies"]
