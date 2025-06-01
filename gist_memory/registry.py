from __future__ import annotations

"""Plugin registries for compression strategies and validation metrics."""

from typing import Dict, Type


class CompressionStrategy:
    """Base interface for text compression strategies."""

    id: str

    def compress(self, text: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class ValidationMetric:
    """Base interface for evaluating predictions."""

    id: str

    def compute(self, reference: str, prediction: str) -> float:  # pragma: no cover
        raise NotImplementedError


_COMPRESSION_REGISTRY: Dict[str, Type[CompressionStrategy]] = {}
_VALIDATION_REGISTRY: Dict[str, Type[ValidationMetric]] = {}


def register_compression_strategy(id: str, strategy_class: Type[CompressionStrategy]) -> None:
    """Register ``strategy_class`` under ``id``."""

    _COMPRESSION_REGISTRY[id] = strategy_class


def register_validation_metric(id: str, metric_class: Type[ValidationMetric]) -> None:
    """Register ``metric_class`` under ``id``."""

    _VALIDATION_REGISTRY[id] = metric_class


__all__ = [
    "CompressionStrategy",
    "ValidationMetric",
    "register_compression_strategy",
    "register_validation_metric",
    "_COMPRESSION_REGISTRY",
    "_VALIDATION_REGISTRY",
]
