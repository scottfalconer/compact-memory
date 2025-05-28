import numpy as np
from datetime import datetime
import pytest
from gist_memory.json_npy_store import JsonNpyVectorStore
from gist_memory.models import BeliefPrototype, RawMemory
from gist_memory.embedding_pipeline import EmbeddingDimensionMismatchError


def test_json_npy_roundtrip(tmp_path):
    store = JsonNpyVectorStore(path=str(tmp_path), embedding_model="test", embedding_dim=3)
    proto = BeliefPrototype(
        prototype_id="p1",
        vector_row_index=0,
        summary_text="test",
        strength=1.0,
        confidence=1.0,
        creation_ts=datetime.utcnow(),
        last_updated_ts=datetime.utcnow(),
    )
    store.add_prototype(proto, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    mem = RawMemory(
        memory_id="m1",
        raw_text_hash="hash",
        assigned_prototype_id="p1",
        raw_text="hello",
        source_document_id=None,
        embedding=[0.1, 0.0, 0.0],
    )
    store.add_memory(mem)
    store.save()

    # reload
    store2 = JsonNpyVectorStore(path=str(tmp_path))
    assert store2.prototypes[0].prototype_id == "p1"
    assert len(store2.memories) == 1
    assert store2.embedding_dim == 3


def test_meta_validation(tmp_path):
    store = JsonNpyVectorStore(path=str(tmp_path), embedding_model="test", embedding_dim=3)
    proto = BeliefPrototype(
        prototype_id="p1",
        vector_row_index=0,
        summary_text="a",
        strength=1.0,
        confidence=1.0,
        creation_ts=datetime.utcnow(),
        last_updated_ts=datetime.utcnow(),
    )
    store.add_prototype(proto, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    store.save()
    meta_path = tmp_path / "meta.yaml"
    import yaml

    meta = yaml.safe_load(meta_path.read_text())
    meta["embedding_dim"] = 2
    meta_path.write_text(yaml.safe_dump(meta))
    with pytest.raises(EmbeddingDimensionMismatchError):
        JsonNpyVectorStore(path=str(tmp_path))

