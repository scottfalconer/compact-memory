from pathlib import Path

from compact_memory.experiment_runner import ExperimentConfig, run_experiment
from compact_memory.embedding_pipeline import MockEncoder
import pytest


@pytest.fixture(autouse=True)
def use_mock_encoder(monkeypatch):
    enc = MockEncoder()
    monkeypatch.setattr(
        "compact_memory.embedding_pipeline._load_model", lambda *a, **k: enc
    )
    yield


def test_run_experiment(tmp_path):
    data = tmp_path / "data.txt"
    data.write_text("hello world")
    cfg = ExperimentConfig(dataset=data, work_dir=tmp_path)
    metrics = run_experiment(cfg)
    assert metrics["memories_ingested"] >= 1
    assert metrics["prototype_count"] >= 1
