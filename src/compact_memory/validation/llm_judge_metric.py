from __future__ import annotations

"""LLM-based evaluation metric using OpenAI chat models."""

from typing import Any, Dict, Optional
import hashlib
import os

import openai

from .metrics_abc import ValidationMetric
from .registry import register_validation_metric

_CACHE: dict[str, float] = {}


class LLMJudgeMetric(ValidationMetric):
    """Score text pairs by querying an OpenAI model."""

    metric_id = "llm_judge"

    DEFAULT_ANSWER_PROMPT = (
        "You are an expert grader. "
        "Given a model's answer and the reference correct answer, "
        "score how correct the model answer is on a scale from 0 to 1. "
        "Only output the numeric score."
    )

    DEFAULT_SUMMARY_PROMPT = (
        "You are an evaluator of summaries. "
        "Given an original text and its compressed summary, "
        "score how well the summary preserves the important information on a scale from 0 to 1. "
        "Only output the numeric score."
    )

    def __init__(self, model_name: str = "gpt-3.5-turbo", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model_name = model_name
        api_key = kwargs.pop("api_key", None) or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set")
        self.client = openai.OpenAI(api_key=api_key)

    def _score_pair(self, system_prompt: str, text_a: str, text_b: str) -> float:
        key = hashlib.sha256(
            "||".join([system_prompt, text_a, text_b]).encode()
        ).hexdigest()
        if key in _CACHE:
            return _CACHE[key]
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Text A:\n{text_a}\n\nText B:\n{text_b}\n\nScore:",
            },
        ]
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
        )
        try:
            score = float(resp.choices[0].message.content.strip())
        except Exception as exc:
            raise RuntimeError(f"Invalid LLM judge response: {resp}") from exc
        _CACHE[key] = score
        return score

    def evaluate(
        self,
        *,
        original_text: Optional[str] = None,
        compressed_text: Optional[str] = None,
        llm_response: Optional[str] = None,
        reference_answer: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        if original_text is not None and compressed_text is not None:
            prompt = self.config_params.get(
                "summary_prompt", self.DEFAULT_SUMMARY_PROMPT
            )
            score = self._score_pair(prompt, original_text, compressed_text)
        elif llm_response is not None and reference_answer is not None:
            prompt = self.config_params.get("answer_prompt", self.DEFAULT_ANSWER_PROMPT)
            score = self._score_pair(prompt, reference_answer, llm_response)
        else:
            raise ValueError(
                "LLMJudgeMetric requires original/compressed texts or response/reference texts."
            )
        return {"llm_judge_score": score}


register_validation_metric(LLMJudgeMetric.metric_id, LLMJudgeMetric)

__all__ = ["LLMJudgeMetric"]
