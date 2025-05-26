import numpy as np
import pytest

from gist_memory import embedder as emb
from gist_memory.embedder import RandomEmbedder


def test_random_embedder_dim_and_seed():
    embedder = RandomEmbedder(dim=32, seed=42)
    vec = embedder.embed("test")
    assert len(vec) == 32

    embedder2 = RandomEmbedder(dim=32, seed=42)
    vec2 = embedder2.embed("test")
    assert (vec == vec2).all()


def test_openai_embedder(monkeypatch):
    result = {"data": [{"embedding": [0.1, 0.2]}]}

    class Dummy:
        @staticmethod
        def create(input, model):
            return result

    monkeypatch.setattr(emb.openai, "Embedding", Dummy)
    e = emb.OpenAIEmbedder(model="dummy")
    vec = e.embed("hi")
    assert np.allclose(vec, np.array([0.1, 0.2], dtype=np.float32))


def test_local_embedder(monkeypatch):
    class DummyModel:
        def encode(self, text):
            return [0.3, 0.4, 0.5]

    called_kwargs = {}

    def fake_sentence_transformer(name, **kwargs):
        called_kwargs.update(kwargs)
        return DummyModel()

    monkeypatch.setattr(emb, "SentenceTransformer", fake_sentence_transformer)
    e = emb.LocalEmbedder(model_name="dummy")
    vec = e.embed("hi")
    assert np.allclose(vec, np.array([0.3, 0.4, 0.5], dtype=np.float32))
    assert called_kwargs.get("local_files_only") is True
