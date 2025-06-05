from pathlib import Path
from compact_memory.prototype_engine import PrototypeEngine
from compact_memory.vector_store import InMemoryVectorStore
from compact_memory.embedding_pipeline import get_embedding_dim
from compact_memory.engines import load_engine


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


def test_load_engine(tmp_path: Path, patch_embedding_model) -> None:
    dim = get_embedding_dim()
    store = InMemoryVectorStore(embedding_dim=dim)
    # Initialize PrototypeEngine with custom config values
    engine_config = {
        'similarity_threshold': 0.77,
        'dedup_cache_size': 250,
        'update_summaries': True,
        'store_embedding_dim': dim, # Used if PrototypeEngine.__init__ creates a default store
        'chunker_id': 'CustomTestChunkerForPrototype' # Example of another config param
    }
    # Pass the store directly, and config separately.
    # PrototypeEngine.__init__ uses the passed store instance.
    # The config values are used to set attributes.
    engine = PrototypeEngine(store, config=engine_config)
    engine.add_memory("lorem ipsum dolor")
    dest = tmp_path / "engine"
    engine.save(dest)

    loaded = load_engine(dest)
    assert isinstance(loaded, PrototypeEngine) # Ensure we loaded the correct type
    res = loaded.query("lorem", top_k_prototypes=1, top_k_memories=1)
    assert res["memories"]

    # Assertions for loaded configuration in the config object
    assert loaded.config is not None
    assert loaded.config.get('similarity_threshold') == 0.77
    assert loaded.config.get('dedup_cache_size') == 250
    assert loaded.config.get('update_summaries') is True
    assert loaded.config.get('chunker_id') == 'CustomTestChunkerForPrototype'
    # store_embedding_dim was for potential store init, not necessarily a direct attribute.

    # Assertions for engine's attributes reflecting loaded configurations
    assert loaded.similarity_threshold == 0.77
    # dedup_cache_size sets the size of an internal _LRUSet instance
    assert loaded._dedup.size == 250
    assert loaded.update_summaries is True

    # Check for files from BaseCompressionEngine.save()
    # (called by PrototypeEngine.save() via super().save())
    assert (dest / "entries.json").exists()
    assert (dest / "embeddings.npy").exists()
    # Files specific to PrototypeEngine are already checked in test_engine_save_load
    assert (dest / "engine_manifest.json").exists()
    assert (dest / "memories.json").exists()
    assert (dest / "vectors.npy").exists()
