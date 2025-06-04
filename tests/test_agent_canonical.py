import numpy as np
from compact_memory.agent import Agent
from compact_memory.memory_store import MemoryStore
from compact_memory.embedding_pipeline import MockEncoder


def test_add_memory_uses_canonical(monkeypatch, tmp_path):
    captured = {}

    def fake_embed(texts):
        # capture the texts passed for embedding
        captured["texts"] = list(texts) if isinstance(texts, (list, tuple)) else [texts]
        return np.zeros((len(captured["texts"]), MockEncoder.dim), dtype=np.float32)

    monkeypatch.setattr("compact_memory.agent.embed_text", fake_embed)

    store = MemoryStore(path=str(tmp_path), embedding_dim=MockEncoder.dim)
    agent = Agent(store)
    agent.add_memory("hello world", who="bob")
    assert captured["texts"][0].startswith("WHO: bob;")
