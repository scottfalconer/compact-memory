import numpy as np
import uuid

import pytest

from gist_memory import agent as ag
from gist_memory.agent import Agent
from gist_memory.embedding_pipeline import MockEncoder, _load_model, embed_text
from gist_memory.json_npy_store import JsonNpyVectorStore
from gist_memory.chunker import SentenceWindowChunker


@pytest.fixture(autouse=True)
def use_mock_encoder(monkeypatch):
    enc = MockEncoder()
    monkeypatch.setattr("gist_memory.embedding_pipeline._load_model", lambda *a, **k: enc)
    yield


def test_duplicate_handling(tmp_path):
    store = JsonNpyVectorStore(path=str(tmp_path), embedding_model="mock", embedding_dim=MockEncoder.dim)
    agent = Agent(store, chunker=SentenceWindowChunker())
    agent.add_memory("alpha")
    dup = agent.add_memory("alpha")
    assert dup[0].get("duplicate") is True
    assert len(store.memories) == 1
    assert store.prototypes[0].strength == 1.0


def test_snap_and_spawn(tmp_path):
    store = JsonNpyVectorStore(path=str(tmp_path), embedding_model="mock", embedding_dim=MockEncoder.dim)
    agent = Agent(store, chunker=SentenceWindowChunker())
    agent.add_memory("alpha")
    res = agent.add_memory("golf")[0]
    assert res["spawned"] is True
    assert len(store.prototypes) == 2
    res2 = agent.add_memory("delta")[0]
    assert res2["spawned"] is False
    assert len(store.prototypes) == 2


def test_initial_summary(tmp_path):
    store = JsonNpyVectorStore(path=str(tmp_path), embedding_model="mock", embedding_dim=MockEncoder.dim)
    agent = Agent(store, chunker=SentenceWindowChunker())
    agent.add_memory("alpha bravo charlie")
    assert store.prototypes[0].summary_text != ""
    assert "alpha" in store.prototypes[0].summary_text


def test_summary_update(tmp_path, monkeypatch):
    store = JsonNpyVectorStore(path=str(tmp_path), embedding_model="mock", embedding_dim=MockEncoder.dim)
    agent = Agent(store, chunker=SentenceWindowChunker(), update_summaries=True)
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
    store = JsonNpyVectorStore(path=str(tmp_path), embedding_model="mock", embedding_dim=MockEncoder.dim)
    agent = Agent(store, chunker=SentenceWindowChunker())
    agent.add_memory("alpha")
    vec = embed_text(["alpha"])[0]
    before = store.find_nearest(vec, k=1)[0]
    store.save()
    store2 = JsonNpyVectorStore(path=str(tmp_path))
    after = store2.find_nearest(vec, k=1)[0]
    assert before[0] == after[0]
    assert abs(before[1] - after[1]) < 1e-6

