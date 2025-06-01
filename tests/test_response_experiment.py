from pathlib import Path
import pytest

from gist_memory.response_experiment import ResponseExperimentConfig, run_response_experiment
from gist_memory.embedding_pipeline import MockEncoder


@pytest.fixture(autouse=True)
def use_mock_encoder(monkeypatch):
    enc = MockEncoder()
    monkeypatch.setattr("gist_memory.response_experiment.MockEncoder", lambda: enc)
    monkeypatch.setattr("gist_memory.embedding_pipeline._load_model", lambda *a, **k: enc)
    yield


def test_response_experiment_runs(monkeypatch, tmp_path):
    data = Path(__file__).parent / "data" / "response_dialogues.yaml"

    class DummyLLM:
        def __init__(self, *a, **k):
            pass

        tokenizer = staticmethod(lambda text, return_tensors=None, truncation=None, max_length=None: {"input_ids": text.split()})
        model = type("M", (), {"config": type("C", (), {"n_positions": 50})()})()
        max_new_tokens = 10

        def prepare_prompt(self, agent, prompt, **kw):
            DummyLLM.prompt = prompt
            return prompt

        def reply(self, prompt):
            if "module" in prompt:
                return "B"
            if "secret" in prompt:
                return "123"
            return ""

    monkeypatch.setattr("gist_memory.local_llm.LocalChatModel", DummyLLM)

    params = [{"config_prompt_num_forced_recent_turns": 1}]
    cfg = ResponseExperimentConfig(
        dataset=data,
        param_grid=params,
        validation_metrics=[{"id": "exact_match", "params": {}}],
    )
    results = run_response_experiment(cfg)
    assert len(results) == 1
    res = results[0]
    assert res["metrics"]["exact_match"]["exact_match"] == 1.0
    assert "avg_prompt_tokens" in res

