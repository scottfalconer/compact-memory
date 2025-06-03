import pytest
import yaml
from pathlib import Path
from typing import Dict, Any

from pydantic import ValidationError

# Models to test
from compact_memory.api_config import (
    EmbeddingConfig,
    ChunkerConfig,
    LLMProviderAPIConfig,
    StrategyConfig,
    MemoryStoreConfig,
    CompactMemoryConfig
)
# from compact_memory.api_exceptions import ConfigurationError # Not used in these tests directly

# Helper function to create a temporary YAML file
def create_temp_yaml(filepath: Path, data: Dict[str, Any]):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        yaml.dump(data, f)

@pytest.fixture
def sample_embedding_config_dict() -> Dict[str, Any]:
    return {"provider": "huggingface", "model_name": "sentence-transformers/all-MiniLM-L6-v2"}

@pytest.fixture
def sample_chunker_config_dict() -> Dict[str, Any]:
    return {"type": "sentence_window", "params": {"window_size": 3}}

@pytest.fixture
def sample_llm_provider_config_dict() -> Dict[str, Any]:
    return {"provider": "local", "model_name": "tiny-gpt2", "generation_kwargs": {"temperature": 0.7}}

@pytest.fixture
def sample_strategy_config_dict(sample_llm_provider_config_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": "fast_summary",
        "strategy_class_id": "SummarizerStrategy", # Assuming such a class ID exists for a strategy
        "params": {"max_tokens": 100},
        "llm_config": sample_llm_provider_config_dict,
        "tokenizer_name": "gpt2"
    }

@pytest.fixture
def sample_memory_store_config_dict() -> Dict[str, Any]:
    return {"type": "default_json_npy", "path": "/tmp/test_cm_store", "params": {"some_store_param": True}}

@pytest.fixture
def sample_compact_memory_config_dict(
    sample_embedding_config_dict: Dict[str, Any],
    sample_chunker_config_dict: Dict[str, Any],
    sample_llm_provider_config_dict: Dict[str, Any],
    sample_strategy_config_dict: Dict[str, Any],
    sample_memory_store_config_dict: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "version": "1.1",
        "default_embedding_config": sample_embedding_config_dict,
        "default_chunker_config": sample_chunker_config_dict,
        "default_llm_provider_config": sample_llm_provider_config_dict,
        "default_tokenizer_name": "gpt2-medium",
        "strategies": {
            "fast_summary_instance": sample_strategy_config_dict
        },
        "memory_store_config": sample_memory_store_config_dict
    }

# --- Tests for individual config models ---

def test_embedding_config_valid(sample_embedding_config_dict: Dict[str, Any]):
    config = EmbeddingConfig(**sample_embedding_config_dict)
    assert config.provider == sample_embedding_config_dict["provider"]
    assert config.model_name == sample_embedding_config_dict["model_name"]

def test_embedding_config_missing_fields(sample_embedding_config_dict: Dict[str, Any]):
    incomplete_data = sample_embedding_config_dict.copy()
    del incomplete_data["model_name"]
    with pytest.raises(ValidationError):
        EmbeddingConfig(**incomplete_data)

def test_chunker_config_valid(sample_chunker_config_dict: Dict[str, Any]):
    config = ChunkerConfig(**sample_chunker_config_dict)
    assert config.type == sample_chunker_config_dict["type"]
    assert config.params["window_size"] == 3

def test_llm_provider_config_valid(sample_llm_provider_config_dict: Dict[str, Any]):
    config = LLMProviderAPIConfig(**sample_llm_provider_config_dict)
    assert config.provider == sample_llm_provider_config_dict["provider"]
    assert config.generation_kwargs["temperature"] == 0.7

def test_strategy_config_valid(sample_strategy_config_dict: Dict[str, Any]):
    config = StrategyConfig(**sample_strategy_config_dict)
    assert config.id == sample_strategy_config_dict["id"]
    assert config.llm_config is not None
    assert config.llm_config.provider == sample_strategy_config_dict["llm_config"]["provider"]

def test_memory_store_config_valid(sample_memory_store_config_dict: Dict[str, Any]):
    config = MemoryStoreConfig(**sample_memory_store_config_dict)
    assert config.type == sample_memory_store_config_dict["type"]
    assert config.path == sample_memory_store_config_dict["path"]

# --- Tests for CompactMemoryConfig ---

def test_compact_memory_config_valid_creation(sample_compact_memory_config_dict: Dict[str, Any]):
    config = CompactMemoryConfig(**sample_compact_memory_config_dict)
    assert config.version == "1.1"
    assert config.default_embedding_config.provider == sample_compact_memory_config_dict["default_embedding_config"]["provider"]
    assert config.strategies["fast_summary_instance"].id == "fast_summary"
    assert config.memory_store_config.path == "/tmp/test_cm_store"

def test_compact_memory_config_missing_required_sub_model(sample_compact_memory_config_dict: Dict[str, Any]):
    invalid_config_data = sample_compact_memory_config_dict.copy()
    del invalid_config_data["default_chunker_config"]
    with pytest.raises(ValidationError):
        CompactMemoryConfig(**invalid_config_data)

def test_compact_memory_config_invalid_strategy_format(sample_compact_memory_config_dict: Dict[str, Any]):
    invalid_config_data = sample_compact_memory_config_dict.copy()
    # Strategies should be a dict, not a list
    invalid_config_data["strategies"] = [sample_compact_memory_config_dict["strategies"]["fast_summary_instance"]]
    with pytest.raises(ValidationError):
        CompactMemoryConfig(**invalid_config_data)

def test_compact_memory_config_optional_llm_provider(sample_compact_memory_config_dict: Dict[str, Any]):
    config_data = sample_compact_memory_config_dict.copy()
    del config_data["default_llm_provider_config"]
    config = CompactMemoryConfig(**config_data)
    assert config.default_llm_provider_config is None


# --- Tests for save_to_file and from_file ---

def test_save_and_load_compact_memory_config(tmp_path: Path, sample_compact_memory_config_dict: Dict[str, Any]):
    config_filepath = tmp_path / "test_config.yaml"

    # Create and save the config
    original_config = CompactMemoryConfig(**sample_compact_memory_config_dict)
    original_config.save_to_file(config_filepath)
    assert config_filepath.exists()

    # Load the config
    loaded_config = CompactMemoryConfig.from_file(config_filepath)

    # Assertions to ensure loaded config matches original
    assert loaded_config.version == original_config.version
    assert loaded_config.default_embedding_config == original_config.default_embedding_config
    assert loaded_config.default_chunker_config == original_config.default_chunker_config
    assert loaded_config.default_llm_provider_config == original_config.default_llm_provider_config
    assert loaded_config.default_tokenizer_name == original_config.default_tokenizer_name
    assert loaded_config.memory_store_config == original_config.memory_store_config

    # Compare strategies (Pydantic models should be comparable directly)
    assert loaded_config.strategies.keys() == original_config.strategies.keys()
    for key in original_config.strategies:
        assert loaded_config.strategies[key] == original_config.strategies[key]

    # A more direct comparison if the models are fully identical
    assert loaded_config == original_config

def test_from_file_non_existent(tmp_path: Path):
    non_existent_filepath = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        CompactMemoryConfig.from_file(non_existent_filepath)

def test_from_file_invalid_yaml(tmp_path: Path):
    invalid_yaml_filepath = tmp_path / "invalid.yaml"
    with open(invalid_yaml_filepath, 'w') as f:
        f.write("default_embedding_config: {provider: 'test', model_name: 'testmodel', invalid_yaml_here::}") # Invalid YAML syntax

    with pytest.raises(yaml.YAMLError): # Or whatever specific error PyYAML raises for parsing
        CompactMemoryConfig.from_file(invalid_yaml_filepath)

def test_from_file_validation_error_on_load(tmp_path: Path, sample_compact_memory_config_dict: Dict[str, Any]):
    config_filepath = tmp_path / "validation_error_config.yaml"
    bad_data = sample_compact_memory_config_dict.copy()
    # Make some data invalid, e.g., missing a required field in a sub-model
    del bad_data["default_embedding_config"]["provider"]
    create_temp_yaml(config_filepath, bad_data)

    with pytest.raises(ValidationError):
        CompactMemoryConfig.from_file(config_filepath)
