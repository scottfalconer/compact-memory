import pytest
from unittest.mock import patch, MagicMock

# Function and models to test
from compact_memory.api_functions import compress_text
from compact_memory.api_models import CompressedMemoryContext, SourceReference
from compact_memory.api_exceptions import StrategyNotFoundError, CompressionError, ConfigurationError

# We'll need to mock the real components that compress_text interacts with
from compact_memory.compression.strategies_abc import CompressionStrategy
from compact_memory.models import CompressedMemory as InternalCompressedMemory # Internal model
from compact_memory.compression.trace import CompressionTrace # Internal model
from compact_memory.token_utils import Tokenizer # Real type
from compact_memory.llm_providers_abc import LLMProvider # Real type


@pytest.fixture
def mock_strategy_class_fixture(): # Removed self, fixtures don't need it
    # This fixture returns a mock strategy *class*
    mock_strategy_cls = MagicMock(spec=CompressionStrategy) # Spec against the ABC

    # Configure what happens when the class is instantiated
    mock_strategy_instance = MagicMock(spec=CompressionStrategy)
    mock_strategy_instance.id = "mock_test_strategy" # Strategies should have an id

    # Mock the 'compress' method of the instance
    # It should return (InternalCompressedMemory, CompressionTrace)
    mock_internal_cm = InternalCompressedMemory(text="compressed_output_from_mock_strategy")
    # Add source_references to mock_internal_cm if needed for a test case
    # Simulating internal source reference objects. These could be dicts or other simple objects.
    ref1_mock = MagicMock()
    ref1_mock.document_id="doc1"
    ref1_mock.chunk_id="ch1"
    ref1_mock.text="ref_text1" # Assuming internal representation has 'text'
    ref1_mock.score=0.9
    ref1_mock.metadata={"orig_index":0}
    mock_internal_cm.source_references = [ref1_mock]

    mock_trace = CompressionTrace(
        strategy_name="mock_test_strategy",
        original_tokens=200,
        compressed_tokens=50,
        processing_ms=25.5,
        llm_input="llm_prompt_if_any",
        llm_output="llm_response_if_any"
    )
    # to_dict method for CompressionTrace might create a dict from its attributes
    mock_trace.to_dict = MagicMock(return_value=vars(mock_trace).copy())
    mock_trace.steps = [{"type": "mock_step", "details": "details here"}]

    mock_strategy_instance.compress = MagicMock(return_value=(mock_internal_cm, mock_trace))

    # When the class is called (instantiated), return our configured instance
    mock_strategy_cls.return_value = mock_strategy_instance
    return mock_strategy_cls

@pytest.fixture
def mock_tokenizer_instance(): # Removed self
    mock_tokenizer = MagicMock(spec=Tokenizer)
    mock_tokenizer.name = "mock_tokenizer_for_test"
    return mock_tokenizer

@pytest.fixture
def mock_llm_provider_instance(): # Removed self
    mock_llm = MagicMock(spec=LLMProvider)
    mock_llm.model_name = "mock_llm_for_test"
    return mock_llm


# --- Test Cases for compress_text ---

@patch('compact_memory.api_functions.get_compression_strategy_class')
def test_compress_text_successful(
    mock_get_strategy_class, # Patched registry function
    mock_strategy_class_fixture, # Our fixture that returns a mock strategy class
    mock_tokenizer_instance,
    mock_llm_provider_instance
):
    mock_get_strategy_class.return_value = mock_strategy_class_fixture

    input_text = "This is a long piece of text to be compressed."
    strategy_class_id = "RegisteredMockStrategyID" # The ID used to look up in registry
    budget = 100
    strategy_params = {"param_key": "param_value"}

    result_context = compress_text(
        text=input_text,
        strategy_class_id=strategy_class_id,
        budget=budget,
        strategy_params=strategy_params,
        tokenizer_instance=mock_tokenizer_instance,
        llm_provider_instance=mock_llm_provider_instance
    )

    # Assert that the strategy class was retrieved from registry
    mock_get_strategy_class.assert_called_once_with(strategy_class_id)

    # Assert that the strategy class was instantiated correctly
    mock_strategy_class_fixture.assert_called_once_with(params=strategy_params)

    # Assert that the strategy instance's compress method was called
    mock_strategy_instance = mock_strategy_class_fixture.return_value
    mock_strategy_instance.compress.assert_called_once_with(
        text_or_chunks=input_text,
        budget=budget,
        tokenizer=mock_tokenizer_instance,
        llm_provider=mock_llm_provider_instance
    )

    # Assert the returned CompressedMemoryContext
    assert isinstance(result_context, CompressedMemoryContext)
    assert result_context.compressed_text == "compressed_output_from_mock_strategy"
    assert result_context.strategy_id_used == "mock_test_strategy" # From the mock instance's id
    assert result_context.budget_info["requested_budget"] == budget
    assert result_context.budget_info["final_tokens"] == 50 # From mock_trace
    assert result_context.processing_time_ms == 25.5 # From mock_trace
    assert result_context.strategy_llm_input == "llm_prompt_if_any"
    assert result_context.strategy_llm_output == "llm_response_if_any"
    assert len(result_context.source_references) == 1
    assert result_context.source_references[0].document_id == "doc1"
    assert result_context.source_references[0].text_snippet == "ref_text1"
    # Accessing .steps directly, assuming to_dict() populates it this way or vars() includes it.
    assert result_context.full_trace["steps"][0]["type"] == "mock_step"


def test_compress_text_empty_input(): # Removed self
    with pytest.raises(ValueError, match="Input text cannot be empty"):
        compress_text("", "any_strategy", 100)

@patch('compact_memory.api_functions.get_compression_strategy_class', side_effect=StrategyNotFoundError("Strategy not in registry"))
def test_compress_text_strategy_not_found(mock_get_strategy_class):
    with pytest.raises(StrategyNotFoundError, match="Strategy not in registry"):
        compress_text("Some text", "unknown_strategy_id", 100)

@patch('compact_memory.api_functions.get_compression_strategy_class')
def test_compress_text_strategy_instantiation_fails(mock_get_strategy_class):
    mock_strategy_cls = MagicMock(side_effect=Exception("Cannot instantiate this strategy"))
    mock_get_strategy_class.return_value = mock_strategy_cls

    with pytest.raises(CompressionError, match="Failed to compress text: Cannot instantiate this strategy"):
        compress_text("Some text", "failing_strategy", 100)


@patch('compact_memory.api_functions.get_compression_strategy_class')
def test_compress_text_strategy_compress_method_fails(mock_get_strategy_class, mock_strategy_class_fixture):
    # Make the compress method of the strategy instance fail
    mock_strategy_instance = mock_strategy_class_fixture.return_value
    mock_strategy_instance.compress.side_effect = Exception("Compression algorithm failed")
    mock_get_strategy_class.return_value = mock_strategy_class_fixture

    with pytest.raises(CompressionError, match="Failed to compress text: Compression algorithm failed"):
        compress_text("Some text", "any_strategy", 100)


@patch('compact_memory.api_functions.get_compression_strategy_class')
def test_compress_text_strategy_requires_llm_not_provided(mock_get_strategy_class, mock_strategy_class_fixture, mock_tokenizer_instance):
    # Simulate a strategy that requires an LLM
    mock_strategy_instance = mock_strategy_class_fixture.return_value
    # Add 'requires_llm' to the spec of the instance if spec was used strictly, or just set it.
    type(mock_strategy_instance).requires_llm = True # Mocking as a class/type property for getattr
    mock_get_strategy_class.return_value = mock_strategy_class_fixture

    with pytest.raises(ConfigurationError, match="Strategy 'mock_test_strategy' requires an LLM provider, but none was provided."):
        compress_text(
            text="Some text",
            strategy_class_id="llm_requiring_strategy",
            budget=100,
            tokenizer_instance=mock_tokenizer_instance,
            llm_provider_instance=None # Explicitly None
        )
    del type(mock_strategy_instance).requires_llm # Clean up mocked attribute


@patch('compact_memory.api_functions.get_compression_strategy_class')
def test_compress_text_strategy_requires_tokenizer_not_provided(mock_get_strategy_class, mock_strategy_class_fixture, mock_llm_provider_instance):
    mock_strategy_instance = mock_strategy_class_fixture.return_value
    type(mock_strategy_instance).requires_tokenizer = True # Mocking as a class/type property
    mock_get_strategy_class.return_value = mock_strategy_class_fixture

    with pytest.raises(ConfigurationError, match="Strategy 'mock_test_strategy' requires a tokenizer, but none was provided."):
        compress_text(
            text="Some text",
            strategy_class_id="tokenizer_requiring_strategy",
            budget=100,
            tokenizer_instance=None, # Explicitly None
            llm_provider_instance=mock_llm_provider_instance
        )
    del type(mock_strategy_instance).requires_tokenizer # Clean up

@patch('compact_memory.api_functions.get_compression_strategy_class')
def test_compress_text_minimal_trace_and_refs(
    mock_get_strategy_class, mock_strategy_class_fixture, mock_tokenizer_instance
):
    # Configure mock strategy to return minimal trace and no source refs
    mock_strategy_instance = mock_strategy_class_fixture.return_value
    minimal_internal_cm = InternalCompressedMemory(text="minimal_compressed")
    minimal_internal_cm.source_references = [] # No source refs from strategy

    minimal_trace = CompressionTrace(strategy_name="mock_test_strategy")
    minimal_trace.to_dict = MagicMock(return_value=vars(minimal_trace).copy()) # Ensure to_dict exists

    mock_strategy_instance.compress = MagicMock(return_value=(minimal_internal_cm, minimal_trace))
    mock_get_strategy_class.return_value = mock_strategy_class_fixture

    input_text = "Single input text, no refs from strategy."
    result_context = compress_text(
        text=input_text,
        strategy_class_id="any_strategy",
        budget=50,
        tokenizer_instance=mock_tokenizer_instance
    )

    assert result_context.compressed_text == "minimal_compressed"
    assert len(result_context.source_references) == 1
    assert result_context.source_references[0].text_snippet == input_text[:200] # Default snippet length
    assert result_context.source_references[0].document_id == "original_input_text" # Corrected default ID
    assert result_context.full_trace is not None
    assert result_context.full_trace['strategy_name'] == "mock_test_strategy"
