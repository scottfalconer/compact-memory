from __future__ import annotations

"""Registry for validation metrics."""

from typing import Dict, Type

from .metrics_abc import ValidationMetric

_VALIDATION_METRIC_REGISTRY: Dict[str, Type[ValidationMetric]] = {}


def register_validation_metric(id: str, metric_class: Type[ValidationMetric]) -> None:
    """Register ``metric_class`` under ``id``."""
    _VALIDATION_METRIC_REGISTRY[id] = metric_class


def get_validation_metric_class(id: str) -> Type[ValidationMetric]:
    """Return the metric class registered under ``id``."""
    return _VALIDATION_METRIC_REGISTRY[id]


def list_validation_metrics() -> list[str]:
    """Return the sorted list of registered metric IDs."""
    return sorted(_VALIDATION_METRIC_REGISTRY.keys())


__all__ = [
    "register_validation_metric",
    "get_validation_metric_class",
    "list_validation_metrics",
    "_VALIDATION_METRIC_REGISTRY",
]
