from pathlib import Path

from compact_memory.engines import BaseCompressionEngine


def test_engine_ingest_and_recall(patch_embedding_model):
    engine = BaseCompressionEngine()
    ids = engine.ingest("A cat sits. A dog runs.")
    assert len(ids) >= 1
    results = engine.recall("cat", top_k=1)
    assert results
    assert "cat" in results[0]["text"]


def test_engine_save_load(tmp_path: Path, patch_embedding_model):
    engine = BaseCompressionEngine()
    engine.ingest("Hello world.")
    engine.save(tmp_path)

    other = BaseCompressionEngine()
    other.load(tmp_path)
    res = other.recall("Hello")
    assert res
    # ensure files exist
    assert (tmp_path / "entries.json").exists()
    assert (tmp_path / "embeddings.npy").exists()
