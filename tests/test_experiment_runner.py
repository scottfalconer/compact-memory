from pathlib import Path

from compact_memory.experiment_runner import ExperimentConfig, run_experiment
import pytest


def test_run_experiment(tmp_path):
    data = tmp_path / "data.txt"
    data.write_text("hello world")
    cfg = ExperimentConfig(dataset=data, work_dir=tmp_path)
    metrics = run_experiment(cfg)
    assert metrics["memories_ingested"] >= 1
    assert metrics["prototype_count"] >= 1
