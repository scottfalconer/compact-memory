import numpy as np

from compact_memory.models import BeliefPrototype
from compact_memory.vector_store import (
    InMemoryVectorStore,
    PersistentFaissVectorStore,
)
import pytest


def test_add_prototype_updates_internal_state():
    store = InMemoryVectorStore(embedding_dim=2)
    proto = BeliefPrototype(prototype_id="p1", vector_row_index=0)
    vec = np.array([1.0, 0.0], dtype=np.float32)

    store.add_prototype(proto, vec)

    assert len(store.prototypes) == 1
    assert store.index["p1"] == 0
    assert np.allclose(store.proto_vectors[0], vec)
    assert store._index_dirty is True

    result = store.find_nearest(vec, k=1)

    assert result == [("p1", pytest.approx(1.0))]
    assert store.faiss_index is not None
    assert store._index_dirty is False


def test_find_nearest_returns_closest_match():
    store = InMemoryVectorStore(embedding_dim=2)
    proto1 = BeliefPrototype(prototype_id="p1", vector_row_index=0)
    proto2 = BeliefPrototype(prototype_id="p2", vector_row_index=1)
    store.add_prototype(proto1, np.array([1.0, 0.0], dtype=np.float32))
    store.add_prototype(proto2, np.array([0.0, 1.0], dtype=np.float32))

    query = np.array([0.8, 0.2], dtype=np.float32)
    result = store.find_nearest(query, k=1)

    assert result[0][0] == "p1"


def test_update_prototype_changes_vector_and_magnitude():
    store = InMemoryVectorStore(embedding_dim=2)
    proto = BeliefPrototype(prototype_id="p1", vector_row_index=0)
    store.add_prototype(proto, np.array([1.0, 0.0], dtype=np.float32))

    magnitude = store.update_prototype(
        "p1", np.array([0.0, 1.0], dtype=np.float32), "m1", alpha=0.5
    )

    expected_change = np.sqrt(0.5)
    assert magnitude == pytest.approx(expected_change)

    updated = store.proto_vectors[store.index["p1"]]
    expected_vec = np.array([0.5, 0.5])
    expected_norm = expected_vec / np.linalg.norm(expected_vec)
    assert np.allclose(updated, expected_norm, atol=1e-6)
    assert store.prototypes[0].strength == 2.0
    assert store.prototypes[0].constituent_memory_ids == ["m1"]


def test_find_nearest_empty_store_returns_empty_list():
    store = InMemoryVectorStore(embedding_dim=2)
    result = store.find_nearest(np.array([1.0, 0.0], dtype=np.float32), k=1)
    assert result == []
    assert store.faiss_index is None


def test_add_memory_records_entry():
    store = InMemoryVectorStore(embedding_dim=2)
    from compact_memory.models import RawMemory

    memory = RawMemory(
        memory_id="m1", raw_text_hash="h", raw_text="t", embedding=[0.0, 1.0]
    )
    store.add_memory(memory)
    assert store.memories == [memory]


def test_persistent_faiss_store_save_load(tmp_path):
    store = PersistentFaissVectorStore(embedding_dim=2, path=tmp_path)
    proto = BeliefPrototype(prototype_id="p1", vector_row_index=0)
    vec = np.array([1.0, 0.0], dtype=np.float32)
    store.add_prototype(proto, vec)
    from compact_memory.models import RawMemory

    store.add_memory(
        RawMemory(memory_id="m1", raw_text_hash="h", raw_text="t", embedding=[1.0, 0.0])
    )
    store.save(tmp_path)

    other = PersistentFaissVectorStore(embedding_dim=2, path=tmp_path)
    other.load(tmp_path)
    result = other.find_nearest(vec, k=1)
    assert result[0][0] == "p1"
    assert len(other.memories) == 1


def test_persistent_store_consistent_with_memory_store(tmp_path):
    mem_store = InMemoryVectorStore(embedding_dim=2)
    pers_store = PersistentFaissVectorStore(embedding_dim=2, path=tmp_path)
    proto = BeliefPrototype(prototype_id="p1", vector_row_index=0)
    vec = np.array([0.0, 1.0], dtype=np.float32)
    mem_store.add_prototype(proto, vec)
    pers_store.add_prototype(proto, vec)

    query = np.array([0.0, 1.0], dtype=np.float32)
    res_mem = mem_store.find_nearest(query, k=1)
    res_pers = pers_store.find_nearest(query, k=1)
    assert res_mem == res_pers
