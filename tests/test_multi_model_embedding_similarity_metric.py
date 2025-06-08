import numpy as np
from compact_memory.validation.embedding_metrics import (
    MultiModelEmbeddingSimilarityMetric,
)


def test_multi_model_embedding_similarity_basic(patch_embedding_model):
    metric = MultiModelEmbeddingSimilarityMetric(model_names=["dummy-model"])
    scores = metric.evaluate(original_text="hello", compressed_text="hello")
    data = scores["embedding_similarity"]["dummy-model"]
    assert np.isclose(data["similarity"], 1.0)
    assert data["token_count"] == 1


def test_multi_model_embedding_similarity_skip_long(patch_embedding_model):
    metric = MultiModelEmbeddingSimilarityMetric(model_names=["dummy-model"])
    long_text = " ".join("t" + str(i) for i in range(200))
    scores = metric.evaluate(original_text=long_text, compressed_text=long_text)
    assert scores["embedding_similarity"] == {}
