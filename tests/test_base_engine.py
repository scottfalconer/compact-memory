from pathlib import Path

from compact_memory.engines import BaseCompressionEngine, load_engine


def test_engine_ingest_and_recall(patch_embedding_model):
    engine = BaseCompressionEngine()
    ids = engine.ingest("A cat sits. A dog runs.")
    assert len(ids) >= 1
    results = engine.recall("cat", top_k=1)
    assert results
    assert "cat" in results[0]["text"]


def test_engine_save_load(tmp_path: Path, patch_embedding_model):
    engine_config = {'my_custom_param': 'custom_value', 'chunker_id': 'SpecificTestChunker'}
    # Initialize with a specific chunker instance if BaseCompressionEngine's __init__
    # uses the chunker object to set default chunker_id in config.
    # from compact_memory.chunker import SentenceWindowChunker
    # test_chunker = SentenceWindowChunker()
    # engine = BaseCompressionEngine(chunker=test_chunker, config=engine_config)
    # If __init__ primarily relies on config for chunker_id, then just passing config is enough.
    # For BaseCompressionEngine, config is passed, and __init__ will store it.
    # If no chunker instance is passed, it will create a default SentenceWindowChunker.
    # The 'chunker_id' from the passed config will be stored in self.config.
    engine = BaseCompressionEngine(config=engine_config)
    engine.ingest("Hello world.")
    engine.save(tmp_path)

    # Test loading with the engine's own load method
    other = BaseCompressionEngine() # Create a new instance
    other.load(tmp_path) # This doesn't load manifest's config into other.config by current design.
                         # other.config would be default. load_engine is the one that uses manifest's config.
    res = other.recall("Hello")
    assert res

    # ensure files exist
    assert (tmp_path / "entries.json").exists()
    assert (tmp_path / "embeddings.npy").exists()
    assert (tmp_path / "engine_manifest.json").exists()

    # Test loading with the generic load_engine function
    loaded = load_engine(tmp_path)
    res2 = loaded.recall("world")
    assert res2

    # Assertions for loaded configuration
    assert loaded.config is not None
    assert loaded.config.get('my_custom_param') == 'custom_value'
    assert loaded.config.get('chunker_id') == 'SpecificTestChunker'

    # Verify that the loaded engine's actual chunker type corresponds to what might be
    # expected if chunker_id was used for re-instantiation.
    # For BaseCompressionEngine, __init__ takes a chunker instance or creates a default one.
    # It does not, by default, re-instantiate a chunker from chunker_id in config.
    # So, loaded.chunker will be the default SentenceWindowChunker unless a chunker instance
    # was part of the config (which it isn't for simple serializable dicts) or if
    # load_engine or BaseCompressionEngine.__init__ was enhanced to do this.
    # The test for now is that the chunker_id is correctly saved and loaded in the config.
    # Example: from compact_memory.chunker import SentenceWindowChunker
    # assert type(loaded.chunker).__name__ == 'SentenceWindowChunker' # Default behavior
