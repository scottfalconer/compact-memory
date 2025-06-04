import json

import pytest

from compact_memory import Agent
from compact_memory.memory_store import MemoryStore
from compact_memory.embedding_pipeline import MockEncoder, _load_model


@pytest.fixture(autouse=True)
def use_mock_encoder(monkeypatch):
    enc = MockEncoder()
    monkeypatch.setattr(
        "compact_memory.embedding_pipeline._load_model", lambda *a, **k: enc
    )
    yield


def test_negation_conflict_logging(tmp_path, monkeypatch):
    store = MemoryStore(path=str(tmp_path), embedding_dim=MockEncoder.dim)
    agent = Agent(store, chunk_fn=None)
    agent.add_memory("Alice is happy.")
    pid = store.prototypes[0].prototype_id
    monkeypatch.setattr(store, "find_nearest", lambda vec, k=1: [(pid, 1.0)])
    agent.add_memory("Alice is not happy.")
    log_path = tmp_path / "conflicts.jsonl"
    assert log_path.exists()
    rows = [
        json.loads(line) for line in log_path.read_text().splitlines() if line.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["prototype_id"] == pid
