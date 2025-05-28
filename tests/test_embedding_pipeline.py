import numpy as np
import importlib

from gist_memory import embedding_pipeline as ep


def test_mock_encoder_determinism():
    enc = ep.MockEncoder()
    v1 = enc.encode("hello")
    v2 = enc.encode("hello")
    assert np.allclose(v1, v2)
    assert v1.shape[0] == enc.dim


def test_embed_text_uses_mock(monkeypatch):
    enc = ep.MockEncoder()
    monkeypatch.setattr(ep, "_load_model", lambda *args, **kwargs: enc)
    vecs = ep.embed_text(["a", "b"])
    assert vecs.shape == (2, enc.dim)
    exp = enc.encode("a")
    exp = exp / np.linalg.norm(exp)
    assert np.allclose(vecs[0], exp)
