import json
import pytest
from compact_memory.agent import Agent
from compact_memory.json_npy_store import JsonNpyVectorStore
from compact_memory.embedding_pipeline import MockEncoder
from compact_memory.chunker import SentenceWindowChunker


@pytest.fixture(autouse=True)
def use_mock_encoder(monkeypatch):
    enc = MockEncoder()
    monkeypatch.setattr(
        "compact_memory.embedding_pipeline._load_model", lambda *a, **k: enc
    )
    yield


def test_numeric_conflict_logged(tmp_path):
    store = JsonNpyVectorStore(
        path=str(tmp_path), embedding_model="mock", embedding_dim=MockEncoder.dim
    )
    agent = Agent(store, chunker=SentenceWindowChunker(), similarity_threshold=0.7)
    agent.add_memory("The event is on January 1")
    agent.add_memory("The event is on January 2")
    log_path = tmp_path / "conflicts.jsonl"
    assert log_path.exists()
    rows = [
        json.loads(line) for line in log_path.read_text().splitlines() if line.strip()
    ]
    assert rows
    assert rows[0]["reason"] == "numeric_mismatch"


def test_negation_conflict_logged(tmp_path):
    store = JsonNpyVectorStore(
        path=str(tmp_path), embedding_model="mock", embedding_dim=MockEncoder.dim
    )
    agent = Agent(store, chunker=SentenceWindowChunker(), similarity_threshold=0.7)
    agent.add_memory("John is available")
    agent.add_memory("John is not available")
    log_path = tmp_path / "conflicts.jsonl"
    rows = [
        json.loads(line) for line in log_path.read_text().splitlines() if line.strip()
    ]
    assert any(r["reason"] == "negation_conflict" for r in rows)
