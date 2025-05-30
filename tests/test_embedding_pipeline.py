import numpy as np
import importlib
import pytest

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


def test_load_model_failure(monkeypatch):
    def raise_err(*a, **k):
        raise OSError("missing")

    monkeypatch.setattr(ep, "SentenceTransformer", raise_err)
    with pytest.raises(RuntimeError) as exc:
        ep._load_model("bad", "cpu")
    msg = str(exc.value)
    assert "download-model" in msg
    assert "bad" in msg
