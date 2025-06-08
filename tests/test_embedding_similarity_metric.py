import numpy as np
from compact_memory.validation.embedding_metrics import (
    EmbeddingSimilarityMetric,
    MultiEmbeddingSimilarityMetric,
)


def test_embedding_similarity_identical(patch_embedding_model):
    metric = EmbeddingSimilarityMetric()
    scores = metric.evaluate(original_text="hello", compressed_text="hello")
    assert np.isclose(scores["semantic_similarity"], 1.0)


def test_embedding_similarity_different(patch_embedding_model):
    metric = EmbeddingSimilarityMetric()
    scores = metric.evaluate(original_text="hello", compressed_text="world")
    assert scores["semantic_similarity"] < 1.0


def test_multi_similarity_returns_per_model_scores(patch_embedding_model, monkeypatch):
    import compact_memory.embedding_pipeline as ep

    enc_a = ep.MockEncoder()
    enc_b = ep.MockEncoder()

    def fake_load(name: str, device: str):
        return {"model_a": enc_a, "model_b": enc_b}[name]

    monkeypatch.setattr(ep, "_load_model", fake_load)
    metric = MultiEmbeddingSimilarityMetric(model_names=["model_a", "model_b"])
    scores = metric.evaluate(original_text="hello", compressed_text="hello")
    assert set(scores) == {
        "semantic_similarity",
        "model_a",
        "model_b",
        "token_count",
    }
    assert scores["token_count"] > 0
    assert np.isclose(scores["semantic_similarity"], 1.0)


def test_multi_similarity_skips_when_too_long(patch_embedding_model, monkeypatch):
    import compact_memory.embedding_pipeline as ep

    class SmallEncoder(ep.MockEncoder):
        model_max_length = 2

    monkeypatch.setattr(ep, "_load_model", lambda *a, **k: SmallEncoder())
    metric = MultiEmbeddingSimilarityMetric(model_names=["small"], max_tokens=10)
    text = "one two three four"
    scores = metric.evaluate(original_text=text, compressed_text=text)
    assert list(scores) == ["token_count"]
    assert scores["token_count"] > SmallEncoder.model_max_length
