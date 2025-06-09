from pathlib import Path
import pytest  # For tmp_path fixture if not already using it
from unittest import mock

from compact_memory.engines import (
    BaseCompressionEngine,
    CompressedMemory,
    CompressionTrace,
    load_engine,
)
from compact_memory.engines.first_last_engine import FirstLastEngine
from compact_memory.engines.no_compression_engine import NoCompressionEngine
from compact_memory.utils import calculate_sha256

# Assuming patch_embedding_model from conftest.py sets up a mock embedding function
# that BaseCompressionEngine will use, providing deterministic embeddings.

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
    assert result_none.engine_config == engine_config
    assert isinstance(result_none.trace, CompressionTrace)
    assert (
        result_none.trace.engine_name == "base_truncate"
    )  # Base engine specific trace name
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
    assert result_with_prev.engine_config == engine_config
    assert isinstance(result_with_prev.trace, CompressionTrace)
    assert result_with_prev.trace.engine_name == "base_truncate"


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
    assert result_full.engine_config == engine_config  # Includes chunker_id by default
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
        assert result_all.engine_config == engine_config
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
    assert (
        loaded.config.get("my_custom_param") == "custom_value"
    )  # This part of the test remains valid
    assert (
        loaded.config.get("chunker_id") == "SpecificTestChunker"
    )  # This part of the test remains valid


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
