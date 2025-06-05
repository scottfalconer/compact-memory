from __future__ import annotations

"""Validation metrics implemented with the Hugging Face ``evaluate`` package."""

from typing import Any, Dict

try:  # ``evaluate`` is optional for lightweight installs
    import evaluate  # type: ignore
except Exception:  # pragma: no cover - optional dependency missing
    evaluate = None  # type: ignore

from .metrics_abc import ValidationMetric
from .registry import register_validation_metric


class HFValidationMetric(ValidationMetric):
    """Base class for metrics that rely on ``evaluate``."""

    def __init__(self, hf_metric_name: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.hf_metric_name = hf_metric_name
        if evaluate is None:
            raise ImportError(
                "The 'evaluate' package is required to use HFValidationMetric."
            )
        try:
            load_args = self.config_params.get("load_args", {})
            self.metric_loader = evaluate.load(self.hf_metric_name, **load_args)
        except Exception as exc:  # pragma: no cover - passthrough
            raise ValueError(
                f"Could not load Hugging Face metric '{self.hf_metric_name}': {exc}"
            )


class RougeHFMetric(HFValidationMetric):
    metric_id = "rouge_hf"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("rouge", **kwargs)

    def evaluate(
        self, llm_response: str, reference_answer: str, **kwargs: Any
    ) -> Dict[str, float]:
        compute_args = {}
        for key in ["rouge_types", "use_stemmer", "newline_sep"]:
            if key in self.config_params:
                compute_args[key] = self.config_params[key]
        return self.metric_loader.compute(
            predictions=[llm_response], references=[reference_answer], **compute_args
        )


class BleuHFMetric(HFValidationMetric):
    metric_id = "bleu_hf"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("bleu", **kwargs)

    def evaluate(
        self, llm_response: str, reference_answer: str, **kwargs: Any
    ) -> Dict[str, float]:
        compute_args = {}
        if "max_order" in self.config_params:
            compute_args["max_order"] = self.config_params["max_order"]
        return self.metric_loader.compute(
            predictions=[llm_response], references=[reference_answer], **compute_args
        )


class MeteorHFMetric(HFValidationMetric):
    metric_id = "meteor_hf"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("meteor", **kwargs)

    def evaluate(
        self, llm_response: str, reference_answer: str, **kwargs: Any
    ) -> Dict[str, float]:
        return self.metric_loader.compute(
            predictions=[llm_response], references=[reference_answer]
        )


class BertScoreHFMetric(HFValidationMetric):
    metric_id = "bertscore_hf"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("bertscore", **kwargs)

    def evaluate(
        self, llm_response: str, reference_answer: str, **kwargs: Any
    ) -> Dict[str, float]:
        compute_args = {}
        for key in ["lang", "model_type"]:
            if key in self.config_params:
                compute_args[key] = self.config_params[key]
        return self.metric_loader.compute(
            predictions=[llm_response], references=[reference_answer], **compute_args
        )


class ExactMatchMetric(ValidationMetric):
    metric_id = "exact_match"

    def evaluate(
        self, llm_response: str, reference_answer: str, **kwargs: Any
    ) -> Dict[str, float]:
        match = float(llm_response.strip() == reference_answer.strip())
        return {"exact_match": match}


# Register metrics
register_validation_metric(RougeHFMetric.metric_id, RougeHFMetric)
register_validation_metric(BleuHFMetric.metric_id, BleuHFMetric)
register_validation_metric(MeteorHFMetric.metric_id, MeteorHFMetric)
register_validation_metric(BertScoreHFMetric.metric_id, BertScoreHFMetric)
register_validation_metric(ExactMatchMetric.metric_id, ExactMatchMetric)

__all__ = [
    "HFValidationMetric",
    "RougeHFMetric",
    "BleuHFMetric",
    "MeteorHFMetric",
    "BertScoreHFMetric",
    "ExactMatchMetric",
]
