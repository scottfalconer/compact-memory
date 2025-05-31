import numpy as np
import uuid

import pytest

from gist_memory import agent as ag
from gist_memory.agent import Agent
from gist_memory.embedding_pipeline import MockEncoder, _load_model, embed_text
from gist_memory.json_npy_store import JsonNpyVectorStore
from gist_memory.chunker import SentenceWindowChunker, Chunker
from gist_memory.active_memory_manager import ActiveMemoryManager
from gist_memory.prompt_budget import PromptBudget
import sys


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


def test_receive_channel_ingest(tmp_path):
    store = JsonNpyVectorStore(path=str(tmp_path), embedding_model="mock", embedding_dim=MockEncoder.dim)
    agent = Agent(store, chunker=SentenceWindowChunker())
    res = agent.receive_channel_message("user", "hello there")
    assert res["action"] == "ingest"
    texts = [m.raw_text for m in agent.store.memories]
    assert "hello there" in texts


def test_receive_channel_query(monkeypatch, tmp_path):
    store = JsonNpyVectorStore(path=str(tmp_path), embedding_model="mock", embedding_dim=MockEncoder.dim)
    agent = Agent(store, chunker=SentenceWindowChunker())
    agent.add_memory("alpha bravo")

    prompts = {}

    class Dummy:
        def __init__(self, *a, **k):
            pass

        def prepare_prompt(self, agent, prompt, **kw):
            prompts["prompt"] = prompt
            return prompt

        def reply(self, prompt):
            return "pong"

    monkeypatch.setattr("gist_memory.local_llm.LocalChatModel", Dummy)
    res = agent.receive_channel_message("user", "alpha?")
    assert res["action"] == "query"
    assert res["reply"] == "pong"
    assert "User asked" in prompts["prompt"]


def test_process_conversational_turn_updates_manager(monkeypatch, tmp_path):
    store = JsonNpyVectorStore(path=str(tmp_path), embedding_model="mock", embedding_dim=MockEncoder.dim)
    agent = Agent(store, chunker=SentenceWindowChunker())

    class Dummy:
        def __init__(self, *a, **k):
            pass

        tokenizer = staticmethod(lambda text, return_tensors=None: {"input_ids": [text.split()]})
        model = type("M", (), {"config": type("C", (), {"n_positions": 50})()})()
        max_new_tokens = 10

        def prepare_prompt(self, agent, prompt, **kw):
            return prompt

        def reply(self, prompt):
            return "resp"

    monkeypatch.setattr("gist_memory.local_llm.LocalChatModel", Dummy)
    mgr = ActiveMemoryManager()
    reply, info = agent.process_conversational_turn("hello?", mgr)
    assert reply == "resp"
    assert len(mgr.history) == 1


def test_prompt_budget_truncates_prompt(monkeypatch, tmp_path):
    store = JsonNpyVectorStore(path=str(tmp_path), embedding_model="mock", embedding_dim=MockEncoder.dim)
    budget = PromptBudget(query=2, recent_history=0, older_history=0, ltm_snippets=0)
    agent = Agent(store, chunker=SentenceWindowChunker(), prompt_budget=budget)

    class Dummy:
        def __init__(self, *a, **k):
            pass

        tokenizer = staticmethod(lambda text, return_tensors=None: {"input_ids": text.split()})
        model = type("M", (), {"config": type("C", (), {"n_positions": 50})()})()
        max_new_tokens = 10

        def prepare_prompt(self, agent, prompt, **kw):
            Dummy.prompt = prompt
            return prompt

        def reply(self, prompt):
            return "resp"

    monkeypatch.setattr("gist_memory.local_llm.LocalChatModel", Dummy)
    mgr = ActiveMemoryManager()
    reply, _ = agent.process_conversational_turn("one two three four five?", mgr)
    assert reply == "resp"
    assert "one two" in Dummy.prompt
    assert "three" not in Dummy.prompt


def test_get_statistics_ephemeral_store(tmp_path):
    store = JsonNpyVectorStore(path=str(tmp_path), embedding_model="mock", embedding_dim=MockEncoder.dim)
    store.path = None
    agent = Agent(store, chunker=SentenceWindowChunker())
    stats = agent.get_statistics()
    assert stats["disk_usage"] == 0


def test_add_memory_tqdm_notebook(monkeypatch, tmp_path):
    store = JsonNpyVectorStore(path=str(tmp_path), embedding_model="mock", embedding_dim=MockEncoder.dim)
    agent = Agent(store, chunker=SentenceWindowChunker())

    updates: list[int] = []

    class DummyBar:
        closed = False

        def __init__(self, *a, **k):
            self.total = k.get("total")

        def update(self, n=1):
            updates.append(n)

        def close(self):
            DummyBar.closed = True

    import types
    monkeypatch.setitem(sys.modules, "tqdm.notebook", types.SimpleNamespace(tqdm=lambda *a, **k: DummyBar(*a, **k)))

    agent.add_memory("alpha bravo", tqdm_notebook=True)

    assert sum(updates) == len(agent.chunker.chunk("alpha bravo"))
    assert DummyBar.closed

class DummyChunker(Chunker):
    id = "dummy"

    def chunk(self, text: str) -> list[str]:
        return [f"dummy:{text}"]


def test_reconfigure_chunker(tmp_path):
    store = JsonNpyVectorStore(path=str(tmp_path), embedding_model="mock", embedding_dim=MockEncoder.dim)
    agent = Agent(store, chunker=SentenceWindowChunker())
    agent.add_memory("alpha")
    agent.chunker = DummyChunker()
    agent.add_memory("bravo")
    assert store.memories[-1].raw_text == "dummy:bravo"
    assert store.meta["chunker"] == "dummy"


def test_reconfigure_similarity_threshold(tmp_path, monkeypatch):
    store = JsonNpyVectorStore(path=str(tmp_path), embedding_model="mock", embedding_dim=MockEncoder.dim)
    agent = Agent(store, chunker=SentenceWindowChunker())
    agent.add_memory("alpha")

    def fake_nearest(vec, k=1):
        return [(store.prototypes[0].prototype_id, 0.6)]

    monkeypatch.setattr(store, "find_nearest", fake_nearest)
    agent.similarity_threshold = 0.5
    res = agent.add_memory("bravo")[0]
    assert res["spawned"] is False
    assert store.meta["tau"] == 0.5

