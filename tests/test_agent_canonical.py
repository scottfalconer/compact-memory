import numpy as np
from gist_memory.agent import Agent
from gist_memory.json_npy_store import JsonNpyVectorStore
from gist_memory.embedding_pipeline import MockEncoder


def test_add_memory_uses_canonical(monkeypatch, tmp_path):
    captured = {}

    def fake_embed(texts):
        # capture the texts passed for embedding
        captured['texts'] = list(texts) if isinstance(texts, (list, tuple)) else [texts]
        return np.zeros((len(captured['texts']), MockEncoder.dim), dtype=np.float32)

    monkeypatch.setattr('gist_memory.agent.embed_text', fake_embed)

    store = JsonNpyVectorStore(path=str(tmp_path), embedding_model='mock', embedding_dim=MockEncoder.dim)
    agent = Agent(store)
    agent.add_memory('hello world', who='bob')
    assert captured['texts'][0].startswith('WHO: bob;')

