import importlib
import builtins
import sys

import pytest


def test_rouge_metric_missing_evaluate(monkeypatch):
    import compact_memory.validation.hf_metrics as hf_metrics

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "evaluate" or name.startswith("evaluate."):
            raise ImportError("missing evaluate")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "evaluate", raising=False)
    importlib.reload(hf_metrics)
    with pytest.raises(ImportError):
        hf_metrics.RougeHFMetric()
    monkeypatch.setattr(builtins, "__import__", original_import)
    importlib.reload(hf_metrics)


def test_embedding_metric_missing_sentence_transformers(
    monkeypatch, patch_embedding_model
):
    import compact_memory.embedding_pipeline as ep
    from compact_memory.validation.embedding_metrics import EmbeddingSimilarityMetric

    # Restore original _load_model function to trigger import
    monkeypatch.setattr(ep, "_load_model", patch_embedding_model)
    ep.unload_model()
    ep.SentenceTransformer = None
    ep._embed_cached.cache_clear()

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sentence_transformers" or name.startswith("sentence_transformers."):
            raise ImportError("missing st")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)

    metric = EmbeddingSimilarityMetric()
    with pytest.raises(ImportError):
        metric.evaluate(original_text="a", compressed_text="b")
    monkeypatch.setattr(builtins, "__import__", original_import)
    importlib.reload(ep)
