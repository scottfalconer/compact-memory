import numpy as np
import uuid

import pytest

from compact_memory import agent as ag
from compact_memory.agent import Agent
from compact_memory.embedding_pipeline import MockEncoder, _load_model, embed_text
from compact_memory.memory_store import MemoryStore
from compact_memory.chunking import ChunkFn


@pytest.fixture(autouse=True)
def use_mock_encoder(monkeypatch):
    enc = MockEncoder()
    monkeypatch.setattr(
        "compact_memory.embedding_pipeline._load_model", lambda *a, **k: enc
    )
    yield


def test_duplicate_handling(tmp_path):
    store = MemoryStore(path=str(tmp_path), embedding_dim=MockEncoder.dim)
    agent = Agent(store, chunk_fn=None)
    agent.add_memory("alpha")
    dup = agent.add_memory("alpha")
    assert dup[0].get("duplicate") is True
    assert len(store.memories) == 1
    assert store.prototypes[0].strength == 1.0


def test_snap_and_spawn(tmp_path):
    store = MemoryStore(path=str(tmp_path), embedding_dim=MockEncoder.dim)
    agent = Agent(store, chunk_fn=None)
    agent.add_memory("alpha")
    res = agent.add_memory("golf")[0]
    assert res["spawned"] is True
    assert len(store.prototypes) == 2
    res2 = agent.add_memory("delta")[0]
    assert res2["spawned"] is False
    assert len(store.prototypes) == 2


def test_initial_summary(tmp_path):
    store = MemoryStore(path=str(tmp_path), embedding_dim=MockEncoder.dim)
    agent = Agent(store, chunk_fn=None)
    agent.add_memory("alpha bravo charlie")
    assert store.prototypes[0].summary_text != ""
    assert "alpha" in store.prototypes[0].summary_text


def test_summary_update(tmp_path, monkeypatch):
    store = MemoryStore(path=str(tmp_path), embedding_dim=MockEncoder.dim)
    agent = Agent(store, chunk_fn=None, update_summaries=True)
    agent.add_memory("alpha bravo")
    first = store.prototypes[0].summary_text

    # force next memory to update the same prototype
    def fake_nearest(vec, k=1):
        return [(store.prototypes[0].prototype_id, 1.0)]

    monkeypatch.setattr(store, "find_nearest", fake_nearest)
    agent.add_memory("alpha delta echo")
    updated = store.prototypes[0].summary_text
    assert updated != ""
    assert updated != first
    assert "delta" in updated


def test_persistence_roundtrip(tmp_path):
    store = MemoryStore(path=str(tmp_path), embedding_dim=MockEncoder.dim)
    agent = Agent(store, chunk_fn=None)
    agent.add_memory("alpha")
    vec = embed_text(["alpha"])[0]
    before = store.find_nearest(vec, k=1)[0]
    store.save()
    store2 = MemoryStore(path=str(tmp_path))
    after = store2.find_nearest(vec, k=1)[0]
    assert before[0] == after[0]
    assert abs(before[1] - after[1]) < 1e-6


def test_receive_channel_removed(tmp_path):
    store = MemoryStore(path=str(tmp_path), embedding_dim=MockEncoder.dim)
    agent = Agent(store, chunk_fn=None)
    assert not hasattr(agent, "receive_channel_message")


def test_process_conversational_turn_removed(tmp_path):
    store = MemoryStore(path=str(tmp_path), embedding_dim=MockEncoder.dim)
    agent = Agent(store, chunk_fn=None)
    assert not hasattr(agent, "process_conversational_turn")


def test_get_statistics_ephemeral_store(tmp_path):
    store = MemoryStore(path=str(tmp_path), embedding_dim=MockEncoder.dim)
    store.path = None
    agent = Agent(store, chunk_fn=None)
    stats = agent.get_statistics()
    assert stats["disk_usage"] == 0


def dummy_chunker(text: str) -> list[str]:
    return [f"dummy:{text}"]


def test_reconfigure_chunker(tmp_path):
    store = MemoryStore(path=str(tmp_path), embedding_dim=MockEncoder.dim)
    agent = Agent(store, chunk_fn=None)
    agent.add_memory("alpha")
    agent.chunk_fn = dummy_chunker
    agent.add_memory("bravo")
    assert store.memories[-1].raw_text == "dummy:bravo"


def test_reconfigure_similarity_threshold(tmp_path, monkeypatch):
    store = MemoryStore(path=str(tmp_path), embedding_dim=MockEncoder.dim)
    agent = Agent(store, chunk_fn=None)
    agent.add_memory("alpha")

    def fake_nearest(vec, k=1):
        return [(store.prototypes[0].prototype_id, 0.6)]

    monkeypatch.setattr(store, "find_nearest", fake_nearest)
    agent.similarity_threshold = 0.5
    res = agent.add_memory("bravo")[0]
    assert res["spawned"] is False
    assert store.meta["tau"] == 0.5
