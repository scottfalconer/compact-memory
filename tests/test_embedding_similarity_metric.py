import numpy as np
from compact_memory.validation.embedding_metrics import EmbeddingSimilarityMetric


def test_embedding_similarity_identical(patch_embedding_model):
    metric = EmbeddingSimilarityMetric()
    scores = metric.evaluate(original_text="hello", compressed_text="hello")
    assert np.isclose(scores["semantic_similarity"], 1.0)


def test_embedding_similarity_different(patch_embedding_model):
    metric = EmbeddingSimilarityMetric()
    scores = metric.evaluate(original_text="hello", compressed_text="world")
    assert scores["semantic_similarity"] < 1.0
