import pytest
from compact_memory.engines.neocortex_transfer import NeocortexTransfer
from compact_memory.engines.base import CompressedMemory, CompressionTrace # Added


@pytest.fixture
def neocortex_engine():
    """Pytest fixture to provide a NeocortexTransfer engine instance."""
    engine = NeocortexTransfer()
    # Optionally, add some prior knowledge for consistent testing
    engine.prior_knowledge = {
        "fox": "A small carnivorous mammal.",
        "moon": "Earth's natural satellite.",
    }
    return engine


def test_engine_initialization(neocortex_engine: NeocortexTransfer):
    """Test that the engine initializes correctly."""
    assert neocortex_engine is not None
    assert neocortex_engine.name == "NeocortexTransfer"
    assert neocortex_engine.long_term_memory_store == []
    assert neocortex_engine.working_memory_context == []
    assert "fox" in neocortex_engine.prior_knowledge


def test_semantic_comprehension_us1(neocortex_engine: NeocortexTransfer):
    """Test User Story 1: Semantic Comprehension."""
    text = "The quick brown fox"
    comprehended_info = neocortex_engine._semantic_comprehension(text)

    assert comprehended_info["original_text"] == text
    assert "quick" in comprehended_info["tokens"]
    assert comprehended_info["main_idea"] == text  # First 5 words, text is shorter
    assert comprehended_info["main_idea"] in neocortex_engine.working_memory_context
    assert (
        "related_prior_knowledge" not in comprehended_info
    )  # "The" is not in prior_knowledge
    assert comprehended_info["comprehension_status"] == "nominal"

    text2 = "fox is a mammal"
    comprehended_info2 = neocortex_engine._semantic_comprehension(text2)
    assert "related_prior_knowledge" in comprehended_info2
    assert (
        comprehended_info2["related_prior_knowledge"] == "A small carnivorous mammal."
    )
    assert len(neocortex_engine.working_memory_context) == 2  # Both main ideas


def test_short_term_retention_us2(neocortex_engine: NeocortexTransfer):
    """Test User Story 2: Short-Term Retention."""
    comprehended_info = {
        "main_idea": "A key concept",
        "tokens": ["A", "key", "concept"],
    }
    retained_info = neocortex_engine._short_term_retention(comprehended_info)

    assert retained_info["chunk_content"] == "A key concept"
    assert retained_info["stm_strength"] == 1.0
    assert retained_info["source_comprehension"] == comprehended_info
    assert retained_info["status"] == "retained_in_stm"


def test_encoding_to_ltm_us3(neocortex_engine: NeocortexTransfer):
    """Test User Story 3: Encoding to Long-Term Memory."""
    retained_info = {
        "chunk_content": "Important fact",
        "stm_strength": 0.8,
        "source_comprehension": {
            "original_text": "This is an important fact to remember.",
            "tokens": ["This", "is", "an", "important", "fact", "to", "remember."],
            "current_context": ["previous idea"],
            "related_prior_knowledge": "contextual info",
        },
    }
    ltm_trace = neocortex_engine._encode_to_long_term_memory(retained_info)

    assert ltm_trace["content"] == "Important fact"
    assert ltm_trace["status"] == "encoded_hippocampal"
    assert ltm_trace["encoding_strength"] == 0.5 * 0.8
    assert (
        ltm_trace["encoding_context"]["original_text"]
        == "This is an important fact to remember."
    )
    assert "contextual info" in ltm_trace["initial_associations"]
    assert ltm_trace["consolidation_level"] == 0.0
    assert ltm_trace["id"] == 0  # First trace
    assert len(neocortex_engine.long_term_memory_store) == 1
    assert neocortex_engine.long_term_memory_store[0] == ltm_trace


def test_consolidation_us4(neocortex_engine: NeocortexTransfer):
    """Test User Story 4: Consolidation and Integration."""
    # Encode a couple of traces
    retained1 = {
        "chunk_content": "fox information",
        "stm_strength": 1.0,
        "source_comprehension": {"original_text": "fox details"},
    }
    trace1 = neocortex_engine._encode_to_long_term_memory(retained1)
    trace1["salience"] = 0.8  # Make it more salient

    retained2 = {
        "chunk_content": "moon facts",
        "stm_strength": 1.0,
        "source_comprehension": {"original_text": "moon details"},
    }
    neocortex_engine._encode_to_long_term_memory(retained2)  # trace_id 1

    retained3 = {
        "chunk_content": "another fox note",
        "stm_strength": 1.0,
        "source_comprehension": {"original_text": "more fox details"},
    }
    neocortex_engine._encode_to_long_term_memory(retained3)  # trace_id 2

    neocortex_engine.trigger_consolidation_phase(
        cycles=5
    )  # Multiple cycles to ensure consolidation

    consolidated_trace1 = neocortex_engine.long_term_memory_store[0]
    consolidated_trace2 = neocortex_engine.long_term_memory_store[1]
    consolidated_trace3 = neocortex_engine.long_term_memory_store[2]

    assert (
        consolidated_trace1["consolidation_level"] > 0.5
    )  # Should have increased significantly
    assert consolidated_trace1["encoding_strength"] > trace1["encoding_strength"]
    if consolidated_trace1["consolidation_level"] >= 0.8:
        assert consolidated_trace1["status"] == "consolidated_cortical"
        # Check linking (trace0 and trace2 both contain "fox")
        assert 2 in consolidated_trace1.get("linked_traces", [])
        assert 0 in consolidated_trace3.get("linked_traces", [])
        assert 0 not in consolidated_trace2.get(
            "linked_traces", []
        )  # trace1 and trace2 no common words


def test_retrieval_us5(neocortex_engine: NeocortexTransfer):
    """Test User Story 5: Retrieval and Reintegration."""
    # Setup: Encode and consolidate some memories
    text1 = "The quick brown fox is quick."
    text2 = "The lazy dog sleeps."
    text3 = "A mission to the moon."

    engine = neocortex_engine  # Use the fixture
    engine.compress(text1)  # id 0
    engine.compress(text2)  # id 1
    engine.compress(text3)  # id 2, "moon" is in prior_knowledge

    engine.trigger_consolidation_phase(cycles=10)  # Consolidate well

    # Test retrieval
    retrieved_fox = engine.decompress("quick fox")
    assert "Content: 'The quick brown fox is'" in retrieved_fox
    assert "ID: 0" in retrieved_fox
    assert "Confidence:" in retrieved_fox
    # Check working memory update
    assert "The quick brown fox is" in engine.working_memory_context

    retrieved_moon = engine.decompress("moon mission")
    assert "Content: 'A mission to the moon'" in retrieved_moon  # main idea
    assert "ID: 2" in retrieved_moon
    assert "Original: 'A mission to the moon.'" in retrieved_moon

    retrieved_dog = engine.decompress("dog")
    assert "Content: 'The lazy dog sleeps'" in retrieved_dog
    assert "ID: 1" in retrieved_dog

    no_match = engine.decompress("unknown alien concept")
    assert "No relevant information found" in no_match


def test_full_flow_compress_consolidate_decompress(neocortex_engine: NeocortexTransfer):
    """Test the full flow: compress -> consolidate -> decompress."""
    text_to_learn = "The diligent student studies cognitive psychology."
    budget = 100 # NeocortexTransfer.compress takes a budget param now

    # Compress (includes US1, US2, US3)
    # compress now returns a CompressedMemory object
    compression_result_cm = neocortex_engine.compress(text_to_learn, budget=budget)

    # Check the CompressedMemory object itself
    assert isinstance(compression_result_cm, CompressedMemory)
    assert compression_result_cm.engine_id == NeocortexTransfer.id
    assert isinstance(compression_result_cm.trace, CompressionTrace)
    assert compression_result_cm.trace.engine_name == NeocortexTransfer.id
    assert compression_result_cm.text == "The diligent student studies cognitive" # First 5 words

    # Assertions on metadata (the old dictionary structure)
    assert compression_result_cm.metadata["trace_status"] == "encoded_hippocampal"
    assert compression_result_cm.metadata["content"] == "The diligent student studies cognitive"


    initial_trace_id = neocortex_engine.long_term_memory_store[0]["id"]

    # Consolidate (US4)
    neocortex_engine.trigger_consolidation_phase(cycles=10)  # Consolidate well

    consolidated_trace = None
    for trace in neocortex_engine.long_term_memory_store:
        if trace["id"] == initial_trace_id:
            consolidated_trace = trace
            break
    assert consolidated_trace is not None
    assert consolidated_trace["status"] == "consolidated_cortical"
    assert consolidated_trace["consolidation_level"] >= 0.8

    # Decompress (US5)
    retrieval_cue = "cognitive psychology"
    decompression_result = neocortex_engine.decompress(retrieval_cue)

    assert f"Content: '{compression_result_cm.text}'" in decompression_result # Use .text from CM
    assert f"ID: {initial_trace_id}" in decompression_result
    assert "Confidence:" in decompression_result
    assert compression_result_cm.text in neocortex_engine.working_memory_context # Use .text


# New dedicated test for compress output structure
def test_neocortex_engine_compress_output_structure(neocortex_engine: NeocortexTransfer):
    text = "This is a test for NeocortexTransfer compress method output."
    budget = 50 # Example budget
    engine_config = neocortex_engine.config # Or a specific test config if needed

    # Test with previous_compression_result = None
    result_none = neocortex_engine.compress(text, budget=budget, previous_compression_result=None)

    assert isinstance(result_none, CompressedMemory)
    expected_text = " ".join(text.split()[:5]).rstrip(".,!?") # five_word_gist logic
    assert result_none.text == expected_text
    assert result_none.engine_id == NeocortexTransfer.id
    assert result_none.engine_config == engine_config # BaseCompressor's config

    assert isinstance(result_none.trace, CompressionTrace)
    assert result_none.trace.engine_name == NeocortexTransfer.id
    assert result_none.trace.strategy_params.get("budget") == budget
    assert result_none.trace.input_summary == {"original_length": len(text)}
    assert result_none.trace.output_summary == {"compressed_length": len(expected_text)}

    assert isinstance(result_none.metadata, dict)
    assert result_none.metadata["content"] == expected_text
    assert "message" in result_none.metadata
    assert "trace_status" in result_none.metadata
    assert "trace_strength" in result_none.metadata


    # Test with a dummy previous_compression_result
    dummy_previous_text = "previous compressed text"
    dummy_previous_trace = CompressionTrace(engine_name="prev_eng", strategy_params={}, input_summary={}, output_summary={})
    previous_result_cm = CompressedMemory(
        text=dummy_previous_text,
        engine_id="prev_eng",
        engine_config={"prev_config": "val"},
        trace=dummy_previous_trace
    )
    result_with_prev = neocortex_engine.compress(text, budget=budget, previous_compression_result=previous_result_cm)

    assert isinstance(result_with_prev, CompressedMemory)
    assert result_with_prev.text == expected_text # NeocortexTransfer might not use previous_compression_result in its current logic
    assert result_with_prev.engine_id == NeocortexTransfer.id
    assert isinstance(result_with_prev.trace, CompressionTrace)
    assert result_with_prev.trace.engine_name == NeocortexTransfer.id
    # Metadata should still reflect the current compression operation
    assert result_with_prev.metadata["content"] == expected_text
