from pathlib import Path
from compact_memory.prototype_engine import PrototypeEngine
from compact_memory.vector_store import InMemoryVectorStore
from compact_memory.embedding_pipeline import get_embedding_dim


def test_engine_save_load(tmp_path: Path, patch_embedding_model) -> None:
    dim = get_embedding_dim()
    store = InMemoryVectorStore(embedding_dim=dim)
    engine = PrototypeEngine(store)
    engine.add_memory("the quick brown fox")
    dest = tmp_path / "engine"
    engine.save(dest)

    other_store = InMemoryVectorStore(embedding_dim=dim)
    other = PrototypeEngine(other_store)
    other.load(dest)
    res = other.query("fox", top_k_prototypes=1, top_k_memories=1)
    assert res["memories"]
    assert (dest / "engine_manifest.json").exists()
    assert (dest / "memories.json").exists()
    assert (dest / "vectors.npy").exists()
    assert (dest / "entries.json").exists()
    assert (dest / "embeddings.npy").exists()
