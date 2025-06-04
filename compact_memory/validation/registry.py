from __future__ import annotations

"""Registry for validation metrics."""

from typing import Dict, Type

from .metrics_abc import ValidationMetric

_VALIDATION_METRIC_REGISTRY: Dict[str, Type[ValidationMetric]] = {}


def register_validation_metric(
    metric_id: str, metric_class: Type[ValidationMetric]
) -> None:
    """Register ``metric_class`` under ``metric_id``."""
    _VALIDATION_METRIC_REGISTRY[metric_id] = metric_class


def get_validation_metric_class(metric_id: str) -> Type[ValidationMetric]:
    """Return the metric class registered under ``metric_id``."""
    return _VALIDATION_METRIC_REGISTRY[metric_id]


__all__ = [
    "register_validation_metric",
    "get_validation_metric_class",
    "_VALIDATION_METRIC_REGISTRY",
]
