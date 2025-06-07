from compact_memory.validation.hf_metrics import (
    BleuHFMetric,
    ExactMatchMetric,
)


def test_exact_match_metric():
    metric = ExactMatchMetric()
    assert (
        metric.evaluate(llm_response="hi", reference_answer="hi")["exact_match"] == 1.0
    )
    assert (
        metric.evaluate(llm_response="hi", reference_answer="bye")["exact_match"] == 0.0
    )


def test_bleu_metric_basic():
    metric = BleuHFMetric()
    scores = metric.evaluate(llm_response="hello world", reference_answer="hello world")
    assert "bleu" in scores
    assert isinstance(scores["bleu"], float)
