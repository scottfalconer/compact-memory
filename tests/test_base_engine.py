from pathlib import Path
import pytest # For tmp_path fixture if not already using it
from unittest import mock

from compact_memory.engines import BaseCompressionEngine, load_engine
from compact_memory.utils import calculate_sha256
# Assuming patch_embedding_model from conftest.py sets up a mock embedding function
# that BaseCompressionEngine will use, providing deterministic embeddings.

def test_engine_ingest_and_recall(patch_embedding_model):
    engine = BaseCompressionEngine()
    # With duplicate detection, if "A cat sits." and "A dog runs." are chunked and compressed
    # to the same value (unlikely for default _compress_chunk), they'd be treated as one.
    # Assuming they are different after _compress_chunk.
    ids = engine.ingest("A cat sits. A dog runs.") # This may create one or more chunks

    # The number of ids depends on the chunker and if chunks are identical after compression
    # For default SentenceWindowChunker and default _compress_chunk:
    # "A cat sits." -> 1 chunk
    # "A dog runs." -> 1 chunk
    # So, 2 chunks if distinct.
    # If "A cat sits. A dog runs." is one chunk, then 1 id.
    # This test should be robust to chunking strategy.
    assert len(ids) >= 1 # At least one chunk ID is returned.

    results = engine.recall("cat", top_k=1)
    assert results
    assert "cat" in results[0]["text"]


def test_engine_save_load(tmp_path: Path, patch_embedding_model):
    engine_config = {'my_custom_param': 'custom_value', 'chunker_id': 'SpecificTestChunker'}
    engine = BaseCompressionEngine(config=engine_config)
    text_to_ingest = "Hello world."
    engine.ingest(text_to_ingest)
    engine.save(tmp_path)

    # Test loading with the engine's own load method
    other = BaseCompressionEngine()
    other.load(tmp_path)
    res = other.recall("Hello")
    assert res
    assert res[0]['text'] == text_to_ingest # Assuming it's the only thing and recalled

    # ensure files exist
    assert (tmp_path / "entries.json").exists()
    assert (tmp_path / "embeddings.npy").exists()
    assert (tmp_path / "engine_manifest.json").exists()

    # Test loading with the generic load_engine function
    loaded = load_engine(tmp_path)
    res2 = loaded.recall("world")
    assert res2
    assert res2[0]['text'] == text_to_ingest

    assert loaded.config is not None
    assert loaded.config.get('my_custom_param') == 'custom_value'
    assert loaded.config.get('chunker_id') == 'SpecificTestChunker'


# --- Tests for SHA256 Duplicate Detection ---

def test_sha256_basic_duplicate_ingestion(patch_embedding_model):
    engine = BaseCompressionEngine()
    text_a = "This is test text A."

    # Ingest text_a for the first time
    ids_a1 = engine.ingest(text_a)
    assert len(ids_a1) == 1 # Assuming text_a results in one chunk
    assert len(engine.memories) == 1
    assert len(engine.memory_hashes) == 1
    assert engine.embeddings.shape[0] == 1
    expected_hash_a = calculate_sha256(text_a) # Assuming _compress_chunk returns text as is
    assert expected_hash_a in engine.memory_hashes

    # Ingest text_a for the second time
    ids_a2 = engine.ingest(text_a)
    assert len(ids_a2) == 0 # No new IDs should be returned for duplicates
    assert len(engine.memories) == 1 # Should still be 1
    assert len(engine.memory_hashes) == 1 # Should still be 1
    assert engine.embeddings.shape[0] == 1 # Should still be 1

def test_sha256_different_texts_ingestion(patch_embedding_model):
    engine = BaseCompressionEngine()
    text_a = "This is test text A."
    text_b = "This is test text B, which is different."

    # Ingest text_a
    engine.ingest(text_a)
    assert len(engine.memories) == 1
    assert len(engine.memory_hashes) == 1
    assert engine.embeddings.shape[0] == 1

    # Ingest text_b
    ids_b = engine.ingest(text_b)
    assert len(ids_b) == 1 # text_b should be new
    assert len(engine.memories) == 2
    assert len(engine.memory_hashes) == 2
    assert engine.embeddings.shape[0] == 2

    hash_a = calculate_sha256(text_a)
    hash_b = calculate_sha256(text_b)
    assert hash_a in engine.memory_hashes
    assert hash_b in engine.memory_hashes

def test_sha256_duplicate_detection_after_load(tmp_path: Path, patch_embedding_model):
    engine_v1 = BaseCompressionEngine()
    text_a = "Persistent text A."
    text_b = "New text B after load."

    # Ingest text_a and save
    engine_v1.ingest(text_a)
    assert len(engine_v1.memories) == 1
    engine_v1.save(tmp_path)

    # Load into a new engine instance
    engine_v2 = BaseCompressionEngine()
    engine_v2.load(tmp_path)

    # Verify hashes are rebuilt on load
    assert len(engine_v2.memory_hashes) == 1
    expected_hash_a = calculate_sha256(text_a) # Assuming _compress_chunk is identity
    assert expected_hash_a in engine_v2.memory_hashes
    assert len(engine_v2.memories) == 1 # Ensure memories are loaded
    assert engine_v2.embeddings.shape[0] == 1 # Ensure embeddings are loaded

    # Ingest text_a again (should be a duplicate)
    ids_a_again = engine_v2.ingest(text_a)
    assert len(ids_a_again) == 0
    assert len(engine_v2.memories) == 1
    assert len(engine_v2.memory_hashes) == 1
    assert engine_v2.embeddings.shape[0] == 1

    # Ingest different text_b (should be new)
    ids_b = engine_v2.ingest(text_b)
    assert len(ids_b) == 1
    assert len(engine_v2.memories) == 2
    assert len(engine_v2.memory_hashes) == 2
    assert engine_v2.embeddings.shape[0] == 2
    expected_hash_b = calculate_sha256(text_b)
    assert expected_hash_b in engine_v2.memory_hashes


# Optional: Test with an engine that modifies the text during _compress_chunk
class ModifyingEngine(BaseCompressionEngine):
    def _compress_chunk(self, chunk_text: str) -> str:
        # Example modification: prefix all chunks
        return "MODIFIED::" + chunk_text

def test_sha256_duplicate_with_modifying_engine(patch_embedding_model):
    engine = ModifyingEngine()
    text_a = "Original text A"
    text_b = "Original text B" # This will also be prefixed by "MODIFIED::"

    # Ingest text_a. It will be stored as "MODIFIED::Original text A"
    engine.ingest(text_a)
    assert len(engine.memories) == 1
    hash_a_modified = calculate_sha256("MODIFIED::" + text_a)
    assert hash_a_modified in engine.memory_hashes

    # Ingest text_a again. Its modified form is already there.
    ids_a_again = engine.ingest(text_a)
    assert len(ids_a_again) == 0
    assert len(engine.memories) == 1

    # Ingest text_b. It will be stored as "MODIFIED::Original text B"
    engine.ingest(text_b)
    assert len(engine.memories) == 2
    hash_b_modified = calculate_sha256("MODIFIED::" + text_b)
    assert hash_b_modified in engine.memory_hashes

    # Now, if we had a text_c that after modification becomes "MODIFIED::Original text A"
    # This is harder to set up without knowing the exact modification.
    # The current ModifyingEngine example is too simple for that.
    # Let's test with a different scenario: two texts that become the same after _compress_chunk

    engine_custom_compress = BaseCompressionEngine()

    # Mock _compress_chunk to make two different inputs result in the same compressed form
    with mock.patch.object(engine_custom_compress, '_compress_chunk', side_effect=lambda t: "SAME_COMPRESSED_OUTPUT" if t in ["Input1", "Input2"] else t):
        engine_custom_compress.ingest("Input1")
        assert len(engine_custom_compress.memories) == 1
        hash_compressed = calculate_sha256("SAME_COMPRESSED_OUTPUT")
        assert hash_compressed in engine_custom_compress.memory_hashes

        # Input2 will be compressed to "SAME_COMPRESSED_OUTPUT", which is a duplicate
        ids_input2 = engine_custom_compress.ingest("Input2")
        assert len(ids_input2) == 0
        assert len(engine_custom_compress.memories) == 1

        # Input3 will be compressed to "Input3" (different)
        engine_custom_compress.ingest("Input3")
        assert len(engine_custom_compress.memories) == 2
        hash_input3 = calculate_sha256("Input3")
        assert hash_input3 in engine_custom_compress.memory_hashes
