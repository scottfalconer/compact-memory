from pathlib import Path
import pytest
from unittest import mock
import tempfile
import shutil
import json
import numpy as np
from pydantic import ValidationError

from compact_memory.engine_config import EngineConfig # Now points to the new Pydantic model
from compact_memory.chunker import FixedSizeChunker # Corrected import
from compact_memory.vector_store import PersistentFaissVectorStore # Corrected import
from compact_memory.embedding_pipeline import embed_text # Corrected import
# If get_embedding_dim is a separate utility:
# from compact_memory.embedding_pipeline import get_embedding_dim # Corrected import path


from compact_memory.engines import (
    BaseCompressionEngine,
    CompressedMemory,
    CompressionTrace,
    # load_engine, # load_engine is now imported from compact_memory.engines.registry # This comment is now misleading
)
# from compact_memory.engines.registry import load_engine # Incorrect import
from compact_memory.engines import load_engine # Corrected import
from compact_memory.engines.first_last_engine import FirstLastEngine
from compact_memory.engines.no_compression_engine import NoCompressionEngine
from compact_memory.utils import calculate_sha256


# Assuming patch_embedding_model from conftest.py sets up a mock embedding function
# that BaseCompressionEngine will use, providing deterministic embeddings.

# Helper function
def save_and_load_engine(engine: BaseCompressionEngine) -> BaseCompressionEngine:
    temp_dir = tempfile.mkdtemp()
    try:
        engine_path = Path(temp_dir) / "test_engine"
        engine.save(str(engine_path))
        loaded_engine = load_engine(str(engine_path))
        return loaded_engine
    finally:
        shutil.rmtree(temp_dir)

# --- Tests for BaseCompressionEngine.compress ---


def test_base_engine_compress_output():
    engine_config = {"test_param": "test_value"}
    engine = BaseCompressionEngine(config=engine_config)
    text = "This is a longer test text for compression."
    budget = 20

    # Test with previous_compression_result = None
    result_none = engine.compress(text, budget, previous_compression_result=None)

    assert isinstance(result_none, CompressedMemory)
    assert result_none.text == text[:budget]  # Base engine truncates
    assert result_none.engine_id == BaseCompressionEngine.id
    # Check that all items from original engine_config are in the result's engine_config
    assert all(item in result_none.engine_config.items() for item in engine_config.items())
    assert isinstance(result_none.trace, CompressionTrace)
    assert result_none.trace.engine_name == BaseCompressionEngine.id # Changed to use self.id
    assert result_none.trace.strategy_params == {"budget": budget}
    assert result_none.trace.input_summary == {"original_length": len(text)}
    assert result_none.trace.output_summary == {"compressed_length": len(text[:budget])}

    # Test with a dummy previous_compression_result
    dummy_previous_text = "previous compressed text"
    dummy_previous_trace = CompressionTrace(
        engine_name="prev_eng", strategy_params={}, input_summary={}, output_summary={}
    )
    previous_result = CompressedMemory(
        text=dummy_previous_text,
        engine_id="prev_eng",
        engine_config={"prev_config": "val"},
        trace=dummy_previous_trace,
    )
    result_with_prev = engine.compress(
        text, budget, previous_compression_result=previous_result
    )

    assert isinstance(result_with_prev, CompressedMemory)
    assert (
        result_with_prev.text == text[:budget]
    )  # Base engine ignores previous_compression_result for its logic
    assert result_with_prev.engine_id == BaseCompressionEngine.id
    # Check that all items from original engine_config are in the result's engine_config
    assert all(item in result_with_prev.engine_config.items() for item in engine_config.items())
    assert isinstance(result_with_prev.trace, CompressionTrace)
    assert result_with_prev.trace.engine_name == BaseCompressionEngine.id # Changed to use self.id


# --- Tests for NoCompressionEngine.compress ---


def test_no_compression_engine_compress_output():
    engine_config = {"custom_cfg": "val"}
    engine = NoCompressionEngine(config=engine_config)
    text = "This is a test text."
    budget_full = 100  # Budget larger than text
    budget_truncate = 10  # Budget smaller than text

    # Test with budget larger than text (no truncation)
    result_full = engine.compress(text, budget_full)
    assert isinstance(result_full, CompressedMemory)
    assert result_full.text == text
    assert result_full.engine_id == NoCompressionEngine.id
    # Check that all items from original engine_config are in the result's engine_config (which is a Pydantic model dump)
    assert all(item in result_full.engine_config.items() for item in engine_config.items())
    assert isinstance(result_full.trace, CompressionTrace)
    assert result_full.trace.engine_name == NoCompressionEngine.id
    assert result_full.trace.strategy_params == {
        "llm_token_budget": budget_full
    }  # Specific to this engine's trace
    assert result_full.trace.input_summary == {"input_length": len(text)}
    assert result_full.trace.output_summary == {"output_length": len(text)}

    # Test with budget smaller than text (truncation)
    result_truncate = engine.compress(text, budget_truncate)
    assert isinstance(result_truncate, CompressedMemory)
    # NoCompressionEngine uses token_utils.truncate_text, which might not be exact char count
    # For simplicity, we check length is less than original, assuming truncation happened.
    # A more precise test would mock tokenizer or use known token counts.
    assert len(result_truncate.text) <= len(text)
    if budget_truncate > 0:  # if budget is 0, text can be empty
        assert (
            len(result_truncate.text) > 0
        )  # Should not be empty if budget > 0 and text is not empty
    assert result_truncate.engine_id == NoCompressionEngine.id
    assert isinstance(result_truncate.trace, CompressionTrace)
    assert result_truncate.trace.engine_name == NoCompressionEngine.id
    assert result_truncate.trace.strategy_params == {
        "llm_token_budget": budget_truncate
    }

    # Test with previous_compression_result
    dummy_previous = CompressedMemory(
        text="prev",
        trace=CompressionTrace(
            engine_name="dummy", strategy_params={}, input_summary={}, output_summary={}
        ),
    )
    result_with_prev = engine.compress(
        text, budget_full, previous_compression_result=dummy_previous
    )
    assert isinstance(result_with_prev, CompressedMemory)
    assert (
        result_with_prev.text == text
    )  # Logic should not be affected by previous_compression_result


# --- Tests for FirstLastEngine.compress ---
# Note: FirstLastEngine uses tiktoken by default if available.
# For robust testing without tiktoken dependency, mock tokenizer or pass a simple one.


def mock_tokenizer(text_input):
    """A simple mock tokenizer that splits by space and optionally decodes."""

    class MockTokenized:
        def __init__(self, tokens):
            self.tokens = tokens

        def __len__(self):
            return len(self.tokens)

    tokens = text_input.split()
    # Simulate tiktoken's behavior for decode if needed by tests
    # For FirstLastEngine, it needs a list of tokens from tokenize_text
    # and then tokenizer.decode(tokens)
    return tokens


def mock_decode(tokens_list):
    return " ".join(tokens_list)


@mock.patch(
    "compact_memory.engines.first_last_engine._DEFAULT_TOKENIZER", None
)  # Ensure tiktoken is not picked up if present
def test_first_last_engine_compress_output():
    engine_config = {"test_cfg": "fle_value"}
    engine = FirstLastEngine(config=engine_config)
    text = "one two three four five six seven eight nine ten"  # 10 words

    # Mock the tokenizer used by FirstLastEngine for deterministic behavior
    # The engine uses compact_memory.token_utils.tokenize_text, which takes the tokenizer
    # and the text. Then it uses tokenizer.decode().

    with mock.patch(
        "compact_memory.token_utils.tokenize_text",
        side_effect=lambda tok, txt: txt.split(),
    ), mock.patch.object(
        engine._chunker,
        "tokenizer",
        create=True,
    ):  # if engine has own tokenizer for some reason
        # This part is tricky as FirstLastEngine gets tokenizer from _DEFAULT_TOKENIZER or kwarg
        # Let's assume we pass it directly or it falls back to split() if _DEFAULT_TOKENIZER is None (mocked above)

        # Test case 1: budget allows all tokens
        budget_all = 10
        result_all = engine.compress(
            text, budget_all, tokenizer=mock_decode
        )  # Pass mock_decode as tokenizer, it will be used for decode
        # tokenize_text will use its default split due to _DEFAULT_TOKENIZER mock

        assert isinstance(result_all, CompressedMemory)
        assert result_all.text == text  # Should keep all
        assert result_all.engine_id == FirstLastEngine.id
        # Check that all items from original engine_config are in the result's engine_config
        assert all(item in result_all.engine_config.items() for item in engine_config.items()) # Corrected indentation
        assert isinstance(result_all.trace, CompressionTrace)
        assert result_all.trace.engine_name == FirstLastEngine.id
        assert result_all.trace.strategy_params == {"llm_token_budget": budget_all}
        assert len(result_all.trace.steps) > 0

        # Test case 2: budget allows first 2 and last 2 (total 4)
        budget_partial = 4  # half = 2
        # For "one two three four five six seven eight nine ten"
        # first 2: "one two"
        # last 2: "nine ten"
        # expected: "one two nine ten"
        expected_partial_text = "one two nine ten"
        result_partial = engine.compress(text, budget_partial, tokenizer=mock_decode)

        assert isinstance(result_partial, CompressedMemory)
        assert result_partial.text == expected_partial_text
        assert result_partial.engine_id == FirstLastEngine.id
        assert isinstance(result_partial.trace, CompressionTrace)
        assert result_partial.trace.strategy_params == {
            "llm_token_budget": budget_partial
        }
        assert result_partial.trace.output_summary["final_tokens"] == 4

        # Test case 3: budget is 0 or None (should keep all if None, or specific behavior for 0)
        # Base engine logic for llm_token_budget=None is to keep all.
        result_none_budget = engine.compress(text, None, tokenizer=mock_decode)
        assert result_none_budget.text == text

        result_zero_budget = engine.compress(text, 0, tokenizer=mock_decode)  # half = 0
        assert (
            result_zero_budget.text == ""
        )  # Keeps tokens[:0] + tokens[-0:] which is empty

        # Test with previous_compression_result
        dummy_previous = CompressedMemory(
            text="prev",
            trace=CompressionTrace(
                engine_name="dummy",
                strategy_params={},
                input_summary={},
                output_summary={},
            ),
        )
        result_with_prev = engine.compress(
            text,
            budget_all,
            previous_compression_result=dummy_previous,
            tokenizer=mock_decode,
        )
        assert isinstance(result_with_prev, CompressedMemory)
        assert (
            result_with_prev.text == text
        )  # Logic should not be affected by previous_compression_result


def test_engine_ingest_and_recall(patch_embedding_model):
    engine = BaseCompressionEngine()
    # With duplicate detection, if "A cat sits." and "A dog runs." are chunked and compressed
    # to the same value (unlikely for default _compress_chunk), they'd be treated as one.
    # Assuming they are different after _compress_chunk.
    ids = engine.ingest("A cat sits. A dog runs.")  # This may create one or more chunks

    # The number of ids depends on the chunker and if chunks are identical after compression
    # For default SentenceWindowChunker and default _compress_chunk:
    # "A cat sits." -> 1 chunk
    # "A dog runs." -> 1 chunk
    # So, 2 chunks if distinct.
    # If "A cat sits. A dog runs." is one chunk, then 1 id.
    # This test should be robust to chunking strategy.
    assert len(ids) >= 1  # At least one chunk ID is returned.

    results = engine.recall("cat", top_k=1)
    assert results
    assert "cat" in results[0]["text"]


def test_engine_save_load(tmp_path: Path, patch_embedding_model):
    engine_config = {
        "my_custom_param": "custom_value",
        "chunker_id": "SpecificTestChunker",
    }
    engine = BaseCompressionEngine(config=engine_config)
    text_to_ingest = "Hello world."
    engine.ingest(text_to_ingest)
    engine.save(tmp_path)

    # Test loading with the engine's own load method
    other = BaseCompressionEngine()
    other.load(tmp_path)
    res = other.recall("Hello")
    assert res
    assert res[0]["text"] == text_to_ingest  # Assuming it's the only thing and recalled

    # ensure files exist
    assert (tmp_path / "entries.json").exists()
    assert (tmp_path / "embeddings.npy").exists()
    assert (tmp_path / "engine_manifest.json").exists()

    # Test loading with the generic load_engine function
    loaded = load_engine(tmp_path)
    res2 = loaded.recall("world")
    assert res2
    assert res2[0]["text"] == text_to_ingest  # This part of the test remains valid

    assert loaded.config is not None  # This part of the test remains valid
    assert loaded.config.model_extra is not None # Ensure model_extra exists
    assert (
        loaded.config.model_extra.get("my_custom_param") == "custom_value"
    )
    assert (
        loaded.config.chunker_id == "SpecificTestChunker" # Direct attribute access
    )


# --- Tests for SHA256 Duplicate Detection (These tests are for ingest, not compress, so should be fine) ---


def test_sha256_basic_duplicate_ingestion(patch_embedding_model):
    engine = BaseCompressionEngine()
    text_a = "This is test text A."

    # Ingest text_a for the first time
    ids_a1 = engine.ingest(text_a)
    assert len(ids_a1) == 1  # Assuming text_a results in one chunk
    assert len(engine.memories) == 1
    assert len(engine.memory_hashes) == 1
    assert engine.embeddings.shape[0] == 1
    expected_hash_a = calculate_sha256(
        text_a
    )  # Assuming _compress_chunk returns text as is
    assert expected_hash_a in engine.memory_hashes

    # Ingest text_a for the second time
    ids_a2 = engine.ingest(text_a)
    assert len(ids_a2) == 0  # No new IDs should be returned for duplicates
    assert len(engine.memories) == 1  # Should still be 1
    assert len(engine.memory_hashes) == 1  # Should still be 1
    assert engine.embeddings.shape[0] == 1  # Should still be 1


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
    assert len(ids_b) == 1  # text_b should be new
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
    expected_hash_a = calculate_sha256(text_a)  # Assuming _compress_chunk is identity
    assert expected_hash_a in engine_v2.memory_hashes
    assert len(engine_v2.memories) == 1  # Ensure memories are loaded
    assert engine_v2.embeddings.shape[0] == 1  # Ensure embeddings are loaded

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
    text_b = "Original text B"  # This will also be prefixed by "MODIFIED::"

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
    with mock.patch.object(
        engine_custom_compress,
        "_compress_chunk",
        side_effect=lambda t: (
            "SAME_COMPRESSED_OUTPUT" if t in ["Input1", "Input2"] else t
        ),
    ):
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


def test_engine_recall_no_memories():
    engine = BaseCompressionEngine()
    results = engine.recall("anything")
    assert results == []


# --- Tests for EngineConfig Integration ---

def test_engine_initialization_with_engine_config(patch_embedding_model):
    """Test BaseCompressionEngine initialization with EngineConfig instance, dict, and kwargs."""
    # 1. With EngineConfig instance
    config_instance = EngineConfig(chunker_id="test_chunker_instance", embedding_dim=128)
    engine1 = BaseCompressionEngine(config=config_instance)
    assert engine1.config == config_instance
    assert engine1.config.chunker_id == "test_chunker_instance"
    assert engine1.config.embedding_dim == 128

    # 2. With a dictionary for config
    config_dict = {"chunker_id": "test_chunker_dict", "vector_store": "in_memory", "embedding_dim": 256} # Changed test_vs_dict
    engine2 = BaseCompressionEngine(config=config_dict)
    assert isinstance(engine2.config, EngineConfig)
    assert engine2.config.chunker_id == "test_chunker_dict"
    assert engine2.config.vector_store == "in_memory" # Changed test_vs_dict
    assert engine2.config.embedding_dim == 256

    # 3. With kwargs
    engine3 = BaseCompressionEngine(chunker_id="test_chunker_kwargs", embedding_dim=512)
    assert isinstance(engine3.config, EngineConfig)
    assert engine3.config.chunker_id == "test_chunker_kwargs"
    assert engine3.config.embedding_dim == 512
    assert engine3.config.vector_store == "in_memory" # Changed to reflect new EngineConfig default

    # 4. Test precedence: kwargs override config dict/instance
    engine4 = BaseCompressionEngine(config={"chunker_id": "A", "embedding_dim": 100}, chunker_id="B")
    assert engine4.config.chunker_id == "B"
    assert engine4.config.embedding_dim == 100 # From dict

    engine5 = BaseCompressionEngine(config=EngineConfig(chunker_id="X", embedding_dim=200), chunker_id="Y", vector_store="Z")
    assert engine5.config.chunker_id == "Y"
    assert engine5.config.vector_store == "Z" # This should be 'Z' (string)
    assert engine5.config.embedding_dim == 200 # From EngineConfig instance


def test_chunker_vectorstore_creation_from_config(tmp_path: Path, patch_embedding_model):
    """Test that chunker and vector_store are created based on EngineConfig."""
    # Test chunker creation
    engine_chunker = BaseCompressionEngine(config=EngineConfig(chunker_id="fixed_size"))
    assert isinstance(engine_chunker.chunker, FixedSizeChunker)

    # Test vector_store creation (and embedding_dim propagation)
    # Need a directory for PersistentFaissVectorStore
    vs_path = tmp_path / "test_vs"
    engine_vs = BaseCompressionEngine(
        config=EngineConfig(
            vector_store="faiss_persistent",
            embedding_dim=128,
            vector_store_path=str(vs_path) # faiss_persistent requires a path
        )
    )
    assert isinstance(engine_vs.vector_store, PersistentFaissVectorStore)
    assert engine_vs.vector_store.embedding_dim == 128
    assert Path(engine_vs.vector_store.path).name == "test_vs"


    # Test that passing chunker_instance overrides config string
    custom_chunker = FixedSizeChunker(size=1000) # Changed chunk_size to size
    engine_custom_chunker = BaseCompressionEngine(
        config=EngineConfig(chunker_id="semantic"), # This should be ignored
        chunker=custom_chunker # Pass as 'chunker' not 'chunker_instance'
    )
    assert engine_custom_chunker.chunker == custom_chunker

    # Test that passing vector_store_instance overrides config string
    custom_vs = PersistentFaissVectorStore(embedding_dim=64, path=str(tmp_path / "custom_vs"))
    engine_custom_vs = BaseCompressionEngine(
        config=EngineConfig(vector_store="in_memory", embedding_dim=32), # These should be ignored (use "in_memory" to align with default)
        vector_store=custom_vs # Pass as 'vector_store' not 'vector_store_instance'
    )
    assert engine_custom_vs.vector_store == custom_vs
    assert engine_custom_vs.vector_store.embedding_dim == 64


# Mock get_embedding_dim for tests where it's called
# Assume the actual embed_text function comes from conftest patch_embedding_model
MOCK_EMBED_TEXT_DIM = 32 # Aligned with conftest.py's patch_embedding_model presumed behavior

@mock.patch("compact_memory.engines.base.get_embedding_dim", return_value=MOCK_EMBED_TEXT_DIM) # Patched where it's used in base.py
def test_embedding_dim_handling_in_init(mock_get_dim, tmp_path: Path, patch_embedding_model):
    """Test embedding_dim handling in __init__ under various scenarios."""

    # 1. Default embedding_fn, no embedding_dim in config -> uses get_embedding_dim()
    # For this, vector_store must be one that doesn't require path, e.g. faiss_memory
    engine1 = BaseCompressionEngine(config=EngineConfig(vector_store="in_memory")) # Changed to in_memory
    assert engine1.vector_store.embedding_dim == MOCK_EMBED_TEXT_DIM
    mock_get_dim.assert_called_once()

    mock_get_dim.reset_mock() # Reset for next case

    # 2. Default embedding_fn, embedding_dim=128 in config -> config overrides
    engine2 = BaseCompressionEngine(config=EngineConfig(vector_store="in_memory", embedding_dim=128)) # Changed to in_memory
    assert engine2.vector_store.embedding_dim == 128
    mock_get_dim.assert_not_called() # Should not be called as dim is provided

    # 3. Custom embedding_fn (mocked, not embed_text), no embedding_dim in config -> ValueError
    custom_embed_fn = mock.Mock()
    # Match the actual error message from BaseCompressionEngine.__init__
    with pytest.raises(ValueError, match="embedding_dim could not be resolved for vector store creation."):
        BaseCompressionEngine(
            embedding_fn=custom_embed_fn,
            config=EngineConfig(vector_store="in_memory") # Changed to in_memory
        )
    mock_get_dim.assert_not_called() # get_embedding_dim is for default fn only

    mock_get_dim.reset_mock()

    # 4. Custom embedding_fn, embedding_dim=64 in config -> config is used
    engine4 = BaseCompressionEngine(
        embedding_fn=custom_embed_fn,
        config=EngineConfig(vector_store="in_memory", embedding_dim=64) # Changed to in_memory
    )
    assert engine4.vector_store.embedding_dim == 64
    mock_get_dim.assert_not_called()


def test_save_and_load_with_engine_config_extras(patch_embedding_model, tmp_path: Path):
    """Test save and load preserves EngineConfig including model_extra fields."""
    original_config = EngineConfig(
        chunker_id="test_chunk_save_load",
        vector_store="in_memory", # Changed to in_memory
        embedding_dim=32, # Aligned with conftest.py's patch_embedding_model presumed behavior
        my_extra_param="extra_value",
        another_extra=42
    )
    engine = BaseCompressionEngine(config=original_config)
    engine.ingest("Some text to enable saving.") # Need some data to save

    loaded_engine = save_and_load_engine(engine)

    assert isinstance(loaded_engine.config, EngineConfig)
    assert loaded_engine.config.chunker_id == "test_chunk_save_load"
    assert loaded_engine.config.vector_store == "in_memory" # Should be "in_memory" as per original_config
    assert loaded_engine.config.embedding_dim == 32
    assert loaded_engine.config.model_extra["my_extra_param"] == "extra_value"
    assert loaded_engine.config.model_extra["another_extra"] == 42
    # Check that the loaded vector_store also got the correct dimension
    assert loaded_engine.vector_store.embedding_dim == 32 # This assumes embed_dim from config is used for VS


def test_embedding_dim_mismatch_warning_on_load(capsys, tmp_path: Path, patch_embedding_model):
    """Test warning for embedding_dim mismatch between config and loaded embeddings.npy."""
    # Assume patch_embedding_model generates embeddings of MOCK_EMBED_TEXT_DIM (e.g., 384)
    # 1. Save an engine with default embedding_fn. Its embeddings.npy will have MOCK_EMBED_TEXT_DIM.
    #    The config will initially have embedding_dim=None, which gets resolved to MOCK_EMBED_TEXT_DIM.
    engine_config_initial = EngineConfig(vector_store="in_memory", embedding_dim=None)
    engine_to_save = BaseCompressionEngine(config=engine_config_initial, embedding_fn=embed_text) # ensure default embed_text is used for get_embedding_dim logic
    engine_to_save.ingest("text to create embeddings")

    # Before saving, the resolved embedding_dim in the config should be MOCK_EMBED_TEXT_DIM (32)
    assert engine_to_save.config.embedding_dim == MOCK_EMBED_TEXT_DIM

    engine_path = tmp_path / "mismatch_test_engine"
    engine_to_save.save(str(engine_path))

    # 2. Manually edit its manifest's config to set embedding_dim: 128
    manifest_path = engine_path / "engine_manifest.json"
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    original_manifest_config_dim = manifest['config']['embedding_dim']
    assert original_manifest_config_dim == MOCK_EMBED_TEXT_DIM # Should have saved the resolved dim

    manifest['config']['embedding_dim'] = 128 # Introduce mismatch
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f)

    # 3. Load the engine via load_engine
    #    Mock get_embedding_dim because load_engine might call it if it re-initializes
    #    the vector store from scratch based on the *modified* config.
    #    However, the crucial part is that it should use the dimension from embeddings.npy
    #    if that file exists.
    with mock.patch("compact_memory.engines.base.get_embedding_dim", return_value=MOCK_EMBED_TEXT_DIM): # Patched where it's used by load_engine path
        loaded_engine = load_engine(str(engine_path))

    # 4. Check capsys.readouterr() for the warning
    captured = capsys.readouterr()
    # The exact warning message from base.py is now:
    # "Warning: EngineConfig embedding_dim ({self.config.embedding_dim}) mismatches effective dimension ({authoritative_embedding_dim}) from manifest/embeddings. Using effective dimension."
    # Test sets manifest config to 128. Authoritative becomes 32 (MOCK_EMBED_TEXT_DIM from .npy).
    expected_warning_part = "Warning: EngineConfig embedding_dim (128) mismatches effective dimension (32) from manifest/embeddings. Using effective dimension."
    assert expected_warning_part in captured.err

    # 5. Verify loaded_engine.vector_store.embedding_dim == MOCK_EMBED_TEXT_DIM (authoritative, from .npy file)
    assert loaded_engine.vector_store.embedding_dim == MOCK_EMBED_TEXT_DIM
    # And the loaded_engine.config.embedding_dim should also be updated to reflect the authoritative dimension
    assert loaded_engine.config.embedding_dim == MOCK_EMBED_TEXT_DIM
