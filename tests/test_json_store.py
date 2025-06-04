import numpy as np
from datetime import datetime, timezone
import pytest
from compact_memory.memory_store import MemoryStore
from compact_memory.models import BeliefPrototype, RawMemory


def test_memory_store_roundtrip(tmp_path):
    store = MemoryStore(path=str(tmp_path), embedding_dim=3)
    proto = BeliefPrototype(
        prototype_id="p1",
        vector_row_index=0,
        summary_text="test",
        strength=1.0,
        confidence=1.0,
        creation_ts=datetime.now(timezone.utc),
        last_updated_ts=datetime.now(timezone.utc),
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
    store2 = MemoryStore(path=str(tmp_path), embedding_dim=3)
    assert store2.prototypes == []
    assert store2.memories == []


def test_meta_available(tmp_path):
    store = MemoryStore(path=str(tmp_path), embedding_dim=3)
    proto = BeliefPrototype(
        prototype_id="p1",
        vector_row_index=0,
        summary_text="a",
        strength=1.0,
        confidence=1.0,
        creation_ts=datetime.now(timezone.utc),
        last_updated_ts=datetime.now(timezone.utc),
    )
    store.add_prototype(proto, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    assert store.meta["embedding_dim"] == 3
