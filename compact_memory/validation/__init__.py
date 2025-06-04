from .metrics_abc import ValidationMetric
from .compression_metrics import CompressionRatioMetric
from .registry import (
    register_validation_metric,
    get_validation_metric_class,
    _VALIDATION_METRIC_REGISTRY,
)

__all__ = [
    "ValidationMetric",
    "CompressionRatioMetric",
    "register_validation_metric",
    "get_validation_metric_class",
    "_VALIDATION_METRIC_REGISTRY",
]

try:  # ``evaluate`` dependency may be missing
    from .hf_metrics import (
        HFValidationMetric,
        RougeHFMetric,
        BleuHFMetric,
        MeteorHFMetric,
        BertScoreHFMetric,
        ExactMatchMetric,
    )
except Exception:  # pragma: no cover - optional dependency may be missing
    HFValidationMetric = RougeHFMetric = BleuHFMetric = MeteorHFMetric = (
        BertScoreHFMetric
    ) = ExactMatchMetric = None  # type: ignore
else:
    __all__ += [
        "HFValidationMetric",
        "RougeHFMetric",
        "BleuHFMetric",
        "MeteorHFMetric",
        "BertScoreHFMetric",
        "ExactMatchMetric",
    ]
