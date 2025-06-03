import sys
import types
import pytest

# Provide a lightweight stub for the optional ``evaluate`` dependency
evaluate_stub = types.SimpleNamespace(load=lambda *a, **k: None)
sys.modules.setdefault("evaluate", evaluate_stub)

from compact_memory.validation.hf_metrics import (
    RougeHFMetric,
    BleuHFMetric,
    MeteorHFMetric,
    BertScoreHFMetric,
    ExactMatchMetric,
    HFValidationMetric,
)
from compact_memory.registry import get_validation_metric_class


def test_exact_match_metric():
    metric = ExactMatchMetric()
    scores = metric.evaluate("hi", "hi")
    assert scores["exact_match"] == 1.0
    scores = metric.evaluate("hi", "bye")
    assert scores["exact_match"] == 0.0


def test_metric_registration_lookup():
    cls = get_validation_metric_class("exact_match")
    assert cls is ExactMatchMetric


def test_hf_metrics(monkeypatch):
    def fake_load(name, **kwargs):
        class FakeMetric:
            def compute(self, predictions, references, **kw):
                return {"score": len(predictions[0]) + len(references[0])}

        fake_load.called = name
        return FakeMetric()

    monkeypatch.setattr("evaluate.load", fake_load)

    metric = RougeHFMetric()
    scores = metric.evaluate("a", "b")
    assert scores["score"] == 2
    assert fake_load.called == "rouge"

    metric = BleuHFMetric()
    metric.evaluate("a", "b")
    assert fake_load.called == "bleu"

    metric = MeteorHFMetric()
    metric.evaluate("a", "b")
    assert fake_load.called == "meteor"

    metric = BertScoreHFMetric()
    metric.evaluate("a", "b")
    assert fake_load.called == "bertscore"


