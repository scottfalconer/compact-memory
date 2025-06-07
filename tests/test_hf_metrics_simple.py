import pytest
from compact_memory.validation.hf_metrics import (
    BleuHFMetric,
    ExactMatchMetric,
    evaluate,
)


def test_exact_match_metric():
    metric = ExactMatchMetric()
    assert (
        metric.evaluate(llm_response="hi", reference_answer="hi")["exact_match"] == 1.0
    )
    assert (
        metric.evaluate(llm_response="hi", reference_answer="bye")["exact_match"] == 0.0
    )


@pytest.mark.skipif(evaluate is None, reason="evaluate package not available")
def test_bleu_metric_basic():
    metric = BleuHFMetric()
    scores = metric.evaluate(llm_response="hello world", reference_answer="hello world")
    assert "bleu" in scores
    assert isinstance(scores["bleu"], float)
