import numpy as np
import importlib
import pytest

from compact_memory import embedding_pipeline as ep


def test_mock_encoder_determinism():
    enc = ep.MockEncoder()
    v1 = enc.encode("hello")
    v2 = enc.encode("hello")
    assert np.allclose(v1, v2)
    assert v1.shape[0] == enc.dim


def test_embed_text_uses_mock():
    enc = ep.MockEncoder()
    vecs = ep.embed_text(["a", "b"])
    assert vecs.shape == (2, enc.dim)
    exp = enc.encode("a")
    exp = exp / np.linalg.norm(exp)
    assert np.allclose(vecs[0], exp)


def test_load_model_failure(monkeypatch, patch_embedding_model):
    def raise_err(*a, **k):
        raise OSError("missing")

    monkeypatch.setattr(ep, "SentenceTransformer", raise_err)
    monkeypatch.setattr(ep, "_load_model", patch_embedding_model)
    with pytest.raises(RuntimeError) as exc:
        ep._load_model("bad", "cpu")
    msg = str(exc.value)
    assert "download-model" in msg
    assert "bad" in msg
    assert "compact-memory download-model" in str(exc.value)


def test_embed_text_optional_preprocess(monkeypatch):
    enc = ep.MockEncoder()

    captured = {}

    def fake_embed(text, *a, **k):
        captured["text"] = text
        return enc.encode(text)

    monkeypatch.setattr(ep, "_embed_cached", fake_embed)
    long_text = "word " * 100

    def pre(t: str) -> str:
        return t.replace("word", "x")

    vec = ep.embed_text(long_text, preprocess_fn=pre)
    assert vec.shape == (enc.dim,)
    assert "x" in captured["text"]
