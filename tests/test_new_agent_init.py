import pytest
from pathlib import Path
import yaml # For creating dummy config files

# Modules to test or use in tests
from compact_memory.new_agent import CompactMemoryAgent
from compact_memory.api_config import CompactMemoryConfig #, EmbeddingConfig, ChunkerConfig, LLMProviderAPIConfig, StrategyConfig, MemoryStoreConfig (CompactMemoryConfig should cover these for dict parsing)
from compact_memory.api_exceptions import ConfigurationError, InitializationError, StrategyNotFoundError, LLMProviderError

# Mock some real components that are normally loaded by agent,
# to avoid heavy model loading in these specific __init__ unit tests.
# We are testing the AGENT'S init logic, not necessarily the full stack of each component here.
from unittest.mock import patch, MagicMock

# Pytest fixture for a sample valid CompactMemoryConfig dictionary
@pytest.fixture
def valid_config_dict(tmp_path): # Using tmp_path for file-based store
    store_path = tmp_path / "test_store"
    return {
        "version": "1.0",
        "default_embedding_config": {
            "provider": "huggingface", # Mocking, real one needs actual model
            "model_name": "mock-hf-embedder"
        },
        "default_chunker_config": {"type": "sentence_window", "params": {"window_size": 3}},
        "default_llm_provider_config": {
            "provider": "local", # Mocking, real one needs actual model
            "model_name": "mock-llm",
            "generation_kwargs": {"temperature": 0.1}
        },
        "default_tokenizer_name": "mock-tokenizer", # Mocking
        "strategies": {
            "test_strat_1": {
                "id": "test_strat_1", # Instance ID
                "strategy_class_id": "MockStrategySimple", # Class ID to be mocked
                "params": {"param1": "value1"}
            },
            "test_strat_2_custom_llm_tokenizer": {
                "id": "test_strat_2", # Corrected id to match key for consistency
                "strategy_class_id": "MockStrategyWithLLM", # Class ID to be mocked
                "params": {},
                "llm_config": {"provider": "local", "model_name": "mock-llm-strategy"},
                "tokenizer_name": "mock-tokenizer-strategy"
            }
        },
        "memory_store_config": {"type": "default_json_npy", "path": str(store_path), "params": {}} # Added params:{}
    }

@pytest.fixture
def mock_embedding_pipeline_fixture(): # Renamed to avoid clash if pytest interprets fixture name with class name
    mock_pipeline = MagicMock()
    mock_pipeline.dimension = 384 # Mock dimension
    return mock_pipeline

@pytest.fixture
def mock_json_npy_store_fixture():  # Renamed
    mock_store = MagicMock()
    return mock_store

@pytest.fixture
def mock_sentence_chunker_fixture(): # Renamed
    return MagicMock()

@pytest.fixture
def mock_llm_provider_fixture(): # Renamed
    mock_provider = MagicMock()
    # Mocking the config attribute that might be accessed by the agent for strategy LLM comparison
    mock_provider.config = MagicMock()
    mock_provider.config.provider = "local" # Matching default_llm_provider_config
    mock_provider.config.model_name = "mock-llm"
    return mock_provider


@pytest.fixture
def mock_tokenizer_fixture(): # Renamed
    mock_tok = MagicMock()
    mock_tok.name = "mock-tokenizer" # Matching default_tokenizer_name
    return mock_tok


@pytest.fixture
def mock_strategy_simple_class_fixture(): # Renamed
    mock_class = MagicMock()
    mock_instance = MagicMock()
    mock_class.return_value = mock_instance
    return mock_class

@pytest.fixture
def mock_strategy_with_llm_class_fixture(): # Renamed
    mock_class = MagicMock()
    mock_instance = MagicMock()
    mock_class.return_value = mock_instance
    return mock_class


# --- Test Cases for __init__ ---

@patch('compact_memory.new_agent.get_embedding_model_info')
@patch('compact_memory.new_agent.EmbeddingPipeline')
@patch('compact_memory.new_agent.JsonNpyVectorStore')
@patch('compact_memory.new_agent.SentenceWindowChunker')
@patch('compact_memory.new_agent.FixedSizeChunker') # Also mock FixedSizeChunker in case config changes
@patch('compact_memory.new_agent.get_llm_provider')
@patch('compact_memory.new_agent.get_tokenizer')
@patch('compact_memory.new_agent.get_compression_strategy_class')
def test_agent_init_successful_with_dict_config(
    mock_get_strategy_class, mock_get_tokenizer, mock_get_llm_provider,
    mock_fixed_size_chunker_cls, mock_sentence_chunker_cls,
    mock_store_cls, mock_embedding_pipeline_cls,
    mock_get_embed_info, valid_config_dict, tmp_path,
    mock_embedding_pipeline_fixture, mock_json_npy_store_fixture, mock_sentence_chunker_fixture, # Use renamed fixtures
    mock_llm_provider_fixture, mock_tokenizer_fixture, mock_strategy_simple_class_fixture, mock_strategy_with_llm_class_fixture
):
    # Setup mocks for components
    mock_get_embed_info.return_value = MagicMock(dimension=384)
    mock_embedding_pipeline_cls.return_value = mock_embedding_pipeline_fixture
    mock_store_cls.return_value = mock_json_npy_store_fixture

    # Chunker selection logic mock
    if valid_config_dict["default_chunker_config"]["type"] == "sentence_window":
        mock_sentence_chunker_cls.return_value = mock_sentence_chunker_fixture
    elif valid_config_dict["default_chunker_config"]["type"] == "fixed_size":
        mock_fixed_size_chunker_cls.return_value = MagicMock() # Or a specific fixed_size_chunker_fixture

    # LLM Provider and Tokenizer mocks (for default and strategy-specific)
    # mock_get_llm_provider needs to return different instances if configs differ for strategies
    # For simplicity, the fixture returns one type of mock. More complex mocking might be needed for deep LLM checks.
    mock_get_llm_provider.return_value = mock_llm_provider_fixture
    mock_get_tokenizer.return_value = mock_tokenizer_fixture

    # Mock strategy class loading based on IDs in config
    def strategy_side_effect(strategy_class_id):
        if strategy_class_id == "MockStrategySimple":
            return mock_strategy_simple_class_fixture
        elif strategy_class_id == "MockStrategyWithLLM":
            return mock_strategy_with_llm_class_fixture
        raise StrategyNotFoundError(f"Test mock strategy {strategy_class_id} not found")
    mock_get_strategy_class.side_effect = strategy_side_effect

    cm_config = CompactMemoryConfig(**valid_config_dict)
    agent = CompactMemoryAgent(config=cm_config, default_strategy_id="test_strat_1")

    assert agent.config == cm_config
    assert agent.embedding_dimension == 384
    mock_embedding_pipeline_cls.assert_called_once()
    mock_store_cls.assert_called_once_with(
        path=str(tmp_path / "test_store"),
        embedding_dim=384,
        embedding_model_name="mock-hf-embedder",
        params={}
    )
    mock_json_npy_store_fixture.load.assert_called_once()

    if valid_config_dict["default_chunker_config"]["type"] == "sentence_window":
        mock_sentence_chunker_cls.assert_called_once_with(**valid_config_dict["default_chunker_config"]["params"])

    assert mock_get_llm_provider.call_count >= 1
    assert mock_get_tokenizer.call_count >= 1
    assert agent.llm_provider == mock_llm_provider_fixture
    assert agent.tokenizer == mock_tokenizer_fixture

    assert "test_strat_1" in agent.strategies
    assert "test_strat_2_custom_llm_tokenizer" in agent.strategies

    # Check call for 'test_strat_1' (MockStrategySimple)
    # It should use the agent's default LLM and tokenizer mocks
    mock_strategy_simple_class_fixture.assert_called_once_with(
        params={"param1": "value1"},
        llm_provider=mock_llm_provider_fixture,
        tokenizer=mock_tokenizer_fixture
    )

    # Check call for 'test_strat_2_custom_llm_tokenizer' (MockStrategyWithLLM)
    # This strategy has its own llm_config and tokenizer_name.
    # mock_get_llm_provider and mock_get_tokenizer will be called for these.
    # The strategy should be instantiated with the results of these calls.
    # We rely on mock_get_llm_provider and mock_get_tokenizer returning distinct mocks if parameters differ,
    # or the same mock_llm_provider_fixture/mock_tokenizer_fixture if parameters are identical to default.
    # For this test, since mock_get_llm_provider.return_value is fixed, it will receive the same mock instance.
    # A more granular test would change the return_value of mock_get_llm_provider based on input config.
    args_strat2, _ = mock_strategy_with_llm_class_fixture.call_args
    assert args_strat2[0]['params'] == {}
    assert args_strat2[0]['llm_provider'] is not None # Should be the one from get_llm_provider for strategy
    assert args_strat2[0]['tokenizer'] is not None # Should be the one from get_tokenizer for strategy


    assert agent.agent_default_strategy_id == "test_strat_1"


@patch('compact_memory.new_agent.get_embedding_model_info')
@patch('compact_memory.new_agent.EmbeddingPipeline')
@patch('compact_memory.new_agent.JsonNpyVectorStore')
# Patch other components as needed for this focused test
@patch('compact_memory.new_agent.SentenceWindowChunker')
@patch('compact_memory.new_agent.get_llm_provider')
@patch('compact_memory.new_agent.get_tokenizer')
@patch('compact_memory.new_agent.get_compression_strategy_class')
def test_agent_init_storage_path_override(
    mock_get_strategy_class, mock_get_tokenizer, mock_get_llm_provider, mock_chunker_cls,
    mock_store_cls, mock_embedding_pipeline_cls, mock_get_embed_info,
    valid_config_dict, tmp_path
):
    mock_get_embed_info.return_value = MagicMock(dimension=384)
    # Mock constructors for other components to avoid side effects
    mock_embedding_pipeline_cls.return_value = MagicMock(dimension=384)
    mock_chunker_cls.return_value = MagicMock()
    mock_get_llm_provider.return_value = MagicMock()
    mock_get_tokenizer.return_value = MagicMock()
    mock_get_strategy_class.return_value = MagicMock() # Returns a mock strategy class


    cm_config = CompactMemoryConfig(**valid_config_dict)
    override_path = tmp_path / "override_store"

    mock_store_instance = MagicMock()
    mock_store_cls.return_value = mock_store_instance

    CompactMemoryAgent(config=cm_config, storage_path=str(override_path))

    mock_store_cls.assert_called_once_with(
        path=str(override_path),
        embedding_dim=384,
        embedding_model_name="mock-hf-embedder",
        params={}
    )
    assert override_path.exists()


def test_agent_init_invalid_config_type(self):
    with pytest.raises(ConfigurationError, match="Invalid config type"):
        CompactMemoryAgent(config="not_a_dict_or_config_object")


def test_agent_init_dict_parse_error(self):
    invalid_dict = {"version": "1.0", "default_embedding_config": "should_be_dict"}
    with pytest.raises(ConfigurationError, match="Failed to parse dictionary into CompactMemoryConfig"):
        CompactMemoryAgent(config=invalid_dict)

@patch('compact_memory.new_agent.get_embedding_model_info', side_effect=Exception("Embedding model info error"))
def test_agent_init_embedding_pipeline_fails(mock_get_info, valid_config_dict):
    cm_config = CompactMemoryConfig(**valid_config_dict)
    with pytest.raises(InitializationError, match="Failed to initialize embedding pipeline"):
        CompactMemoryAgent(config=cm_config)

@patch('compact_memory.new_agent.get_embedding_model_info')
@patch('compact_memory.new_agent.EmbeddingPipeline')
@patch('compact_memory.new_agent.JsonNpyVectorStore', side_effect=Exception("Store init error"))
def test_agent_init_store_fails(mock_store_cls, mock_emb_pipeline_cls, mock_get_info, valid_config_dict):
    mock_get_info.return_value = MagicMock(dimension=384)
    mock_emb_pipeline_cls.return_value = MagicMock(dimension=384)
    cm_config = CompactMemoryConfig(**valid_config_dict)
    with pytest.raises(InitializationError, match="Failed to initialize JsonNpyVectorStore"):
        CompactMemoryAgent(config=cm_config)

@patch('compact_memory.new_agent.get_embedding_model_info')
@patch('compact_memory.new_agent.EmbeddingPipeline')
@patch('compact_memory.new_agent.JsonNpyVectorStore')
def test_agent_init_unsupported_chunker(mock_store_cls, mock_emb_cls, mock_get_info, valid_config_dict):
    mock_get_info.return_value = MagicMock(dimension=384)
    mock_emb_cls.return_value = MagicMock(dimension=384)
    mock_store_cls.return_value = MagicMock() # Store needs to be successfully mocked

    config_dict_bad_chunker = valid_config_dict.copy()
    config_dict_bad_chunker["default_chunker_config"]["type"] = "unknown_chunker"
    cm_config = CompactMemoryConfig(**config_dict_bad_chunker)

    with pytest.raises(ConfigurationError, match="Unsupported chunker type: unknown_chunker"):
        CompactMemoryAgent(config=cm_config)


@patch('compact_memory.new_agent.get_embedding_model_info')
@patch('compact_memory.new_agent.EmbeddingPipeline')
@patch('compact_memory.new_agent.JsonNpyVectorStore')
@patch('compact_memory.new_agent.SentenceWindowChunker')
@patch('compact_memory.new_agent.get_llm_provider', side_effect=LLMProviderError("LLM init failed"))
def test_agent_init_default_llm_fails(mock_get_llm, mock_chunker, mock_store, mock_emb, mock_get_info, valid_config_dict):
    mock_get_info.return_value = MagicMock(dimension=384)
    mock_emb.return_value = MagicMock(dimension=384)
    mock_store.return_value = MagicMock()
    mock_chunker.return_value = MagicMock()
    cm_config = CompactMemoryConfig(**valid_config_dict)
    with pytest.raises(LLMProviderError, match="Failed to initialize default LLM provider .*LLM init failed"): # Adjusted match
        CompactMemoryAgent(config=cm_config)


@patch('compact_memory.new_agent.get_embedding_model_info')
@patch('compact_memory.new_agent.EmbeddingPipeline')
@patch('compact_memory.new_agent.JsonNpyVectorStore')
@patch('compact_memory.new_agent.SentenceWindowChunker')
@patch('compact_memory.new_agent.get_llm_provider')
@patch('compact_memory.new_agent.get_tokenizer')
@patch('compact_memory.new_agent.get_compression_strategy_class', side_effect=StrategyNotFoundError("Test: Strategy class not found"))
def test_agent_init_strategy_class_not_found(mock_get_strat_cls, mock_get_tokenizer, mock_get_llm, mock_chunker, mock_store, mock_emb, mock_get_info, valid_config_dict):
    mock_get_info.return_value = MagicMock(dimension=384)
    mock_emb.return_value = MagicMock(dimension=384)
    mock_store.return_value = MagicMock()
    mock_chunker.return_value = MagicMock()
    mock_get_llm.return_value = MagicMock()
    mock_get_tokenizer.return_value = MagicMock()
    cm_config = CompactMemoryConfig(**valid_config_dict)
    with pytest.raises(InitializationError, match="Could not find strategy class for id 'MockStrategySimple'"):
        CompactMemoryAgent(config=cm_config)

@patch('compact_memory.new_agent.get_embedding_model_info')
@patch('compact_memory.new_agent.EmbeddingPipeline')
@patch('compact_memory.new_agent.JsonNpyVectorStore')
@patch('compact_memory.new_agent.SentenceWindowChunker')
@patch('compact_memory.new_agent.get_llm_provider')
@patch('compact_memory.new_agent.get_tokenizer')
@patch('compact_memory.new_agent.get_compression_strategy_class')
def test_agent_init_strategy_instantiation_fails(
    mock_get_strategy_class, mock_get_tokenizer, mock_get_llm_provider,
    mock_chunker_cls, mock_store_cls, mock_embedding_pipeline_cls,
    mock_get_embed_info, valid_config_dict
):
    mock_get_embed_info.return_value = MagicMock(dimension=384)
    mock_embedding_pipeline_cls.return_value = MagicMock(dimension=384)
    mock_store_cls.return_value = MagicMock()
    mock_chunker_cls.return_value = MagicMock()
    mock_get_llm_provider.return_value = MagicMock()
    mock_get_tokenizer.return_value = MagicMock()

    failing_strategy_mock_class = MagicMock(side_effect=Exception("Strategy instantiation error"))

    def strategy_side_effect(strategy_class_id):
        if strategy_class_id == "MockStrategySimple":
            return failing_strategy_mock_class
        elif strategy_class_id == "MockStrategyWithLLM":
            return MagicMock()
        raise StrategyNotFoundError(f"Test mock strategy {strategy_class_id} not found") # Should not happen with valid_config_dict
    mock_get_strategy_class.side_effect = strategy_side_effect

    cm_config = CompactMemoryConfig(**valid_config_dict)
    with pytest.raises(InitializationError, match="Failed to instantiate strategy 'test_strat_1' of type 'MockStrategySimple'"):
        CompactMemoryAgent(config=cm_config)


def test_agent_init_default_strategy_id_not_in_config(valid_config_dict, caplog):
    cm_config = CompactMemoryConfig(**valid_config_dict)
    # Temporarily remove mocks for components that would be initialized before strategy check
    # to ensure the warning is logged from the intended part of __init__.
    # This test focuses on the default_strategy_id warning.
    with patch('compact_memory.new_agent.EmbeddingPipeline'), \
         patch('compact_memory.new_agent.JsonNpyVectorStore'), \
         patch('compact_memory.new_agent.SentenceWindowChunker'), \
         patch('compact_memory.new_agent.get_llm_provider'), \
         patch('compact_memory.new_agent.get_tokenizer'), \
         patch('compact_memory.new_agent.get_compression_strategy_class'), \
         patch('compact_memory.new_agent.get_embedding_model_info', return_value=MagicMock(dimension=384)):
        CompactMemoryAgent(config=cm_config, default_strategy_id="non_existent_strategy")
    assert "Default strategy_id 'non_existent_strategy' provided but not found" in caplog.text
