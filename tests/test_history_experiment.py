from pathlib import Path

from compact_memory.history_experiment import HistoryExperimentConfig, run_history_experiment
from compact_memory.embedding_pipeline import MockEncoder
import pytest


@pytest.fixture(autouse=True)
def use_mock_encoder(monkeypatch):
    enc = MockEncoder()
    monkeypatch.setattr("compact_memory.history_experiment.MockEncoder", lambda: enc)
    yield


def test_history_experiment_runs(tmp_path):
    data = Path(__file__).parent / "data" / "history_dialogues.yaml"
    params = [
        {
            "config_prompt_num_forced_recent_turns": 1,
            "config_prompt_max_activated_older_turns": 1,
            "config_relevance_boost_factor": 1.0,
        },
        {
            "config_prompt_num_forced_recent_turns": 2,
            "config_prompt_max_activated_older_turns": 2,
            "config_relevance_boost_factor": 1.5,
        },
    ]
    cfg = HistoryExperimentConfig(dataset=data, param_grid=params)
    results = run_history_experiment(cfg)
    assert len(results) == 2
    for res in results:
        assert 0.0 <= res["hit_rate"] <= 1.0
