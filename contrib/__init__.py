"""Optional community-contributed strategies."""

from __future__ import annotations

from compact_memory.compression import register_compression_strategy


def enable_all_contrib_strategies() -> None:
    """Register all strategies contained in :mod:`contrib`."""
    from .strategies.chained import ChainedStrategy
    from .strategies.rationale_episode import RationaleEpisodeStrategy
    from .strategies.prototype_system import PrototypeSystemStrategy

    register_compression_strategy(ChainedStrategy.id, ChainedStrategy, source="contrib")
    register_compression_strategy(
        RationaleEpisodeStrategy.id, RationaleEpisodeStrategy, source="contrib"
    )
    register_compression_strategy(
        PrototypeSystemStrategy.id, PrototypeSystemStrategy, source="contrib"
    )


__all__ = ["enable_all_contrib_strategies"]
