from pathlib import Path

from gist_memory.experiments.config import ExperimentConfig
from gist_memory.experiment_runner import run_experiment
from gist_memory.active_memory_manager import ActiveMemoryManager
from gist_memory.utils import load_agent
from gist_memory.embedding_pipeline import MockEncoder
import pytest


@pytest.fixture(autouse=True)
def use_mock_encoder(monkeypatch):
    enc = MockEncoder()
    monkeypatch.setattr("gist_memory.embedding_pipeline._load_model", lambda *a, **k: enc)
    yield


def test_run_experiment(tmp_path):
    data = tmp_path / "data.txt"
    data.write_text("hello world")
    cfg = ExperimentConfig(dataset=data, work_dir=tmp_path)
    metrics = run_experiment(cfg)
    assert metrics["memories_ingested"] >= 1
    assert metrics["prototype_count"] >= 1


def test_experiment_can_override_default_active_memory_manager_params(tmp_path):
    data = tmp_path / "data.txt"
    data.write_text("hello world")
    params = {"config_max_history_buffer_turns": 7}
    cfg = ExperimentConfig(
        dataset=data, work_dir=tmp_path, active_memory_params=params
    )
    run_experiment(cfg)
    import yaml

    meta = yaml.safe_load((tmp_path / "meta.yaml").read_text())
    assert meta["config_max_history_buffer_turns"] == 7


def test_agent_in_experiment_uses_overridden_active_memory_params(tmp_path):
    data = tmp_path / "data.txt"
    data.write_text("hello world")
    params = {"config_initial_activation": 0.42}
    cfg = ExperimentConfig(
        dataset=data, work_dir=tmp_path, active_memory_params=params
    )
    run_experiment(cfg)
    agent = load_agent(tmp_path)
    meta_params = {k: v for k, v in agent.store.meta.items() if k.startswith("config_")}
    mgr = ActiveMemoryManager(**meta_params)
    assert mgr.config_initial_activation == pytest.approx(0.42)
