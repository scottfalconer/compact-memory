from .metrics_abc import ValidationMetric
from .compression_metrics import CompressionRatioMetric

__all__ = ["ValidationMetric", "CompressionRatioMetric"]

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

try:  # embedding dependencies may be missing
    from .embedding_metrics import EmbeddingSimilarityMetric
except Exception:  # pragma: no cover - optional dependency may be missing
    EmbeddingSimilarityMetric = None  # type: ignore
else:
    __all__ += ["EmbeddingSimilarityMetric"]

try:
    from .llm_judge_metric import LLMJudgeMetric
except Exception:  # pragma: no cover - optional dependency may be missing
    LLMJudgeMetric = None  # type: ignore
else:
    __all__ += ["LLMJudgeMetric"]
