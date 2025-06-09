import pytest
from pydantic import ValidationError
from compact_memory.engine_config import EngineConfig # Now points to the new Pydantic model

def test_engine_config_default_values():
    """Test EngineConfig creation with default values."""
    config = EngineConfig()
    assert config.chunker_id == "fixed_size"
    assert config.vector_store == "faiss_memory"
    assert config.embedding_dim is None
    assert config.model_extra == {}

def test_engine_config_custom_values():
    """Test EngineConfig creation with custom valid values."""
    config = EngineConfig(
        chunker_id="semantic",
        vector_store="faiss_persistent",
        embedding_dim=128
    )
    assert config.chunker_id == "semantic"
    assert config.vector_store == "faiss_persistent"
    assert config.embedding_dim == 128
    assert config.model_extra == {}

def test_engine_config_extra_fields():
    """Test EngineConfig creation with extra fields."""
    config = EngineConfig(my_param=123, another_param="test")
    assert config.model_extra["my_param"] == 123
    assert config.model_extra["another_param"] == "test"
    assert config.chunker_id == "fixed_size" # Check default for non-extra field

def test_engine_config_invalid_type_known_field():
    """Test EngineConfig creation with an invalid type for a known field."""
    with pytest.raises(ValidationError):
        EngineConfig(chunker_id=123)

    with pytest.raises(ValidationError):
        EngineConfig(vector_store=456)

    with pytest.raises(ValidationError):
        EngineConfig(embedding_dim="not_an_int")

def test_engine_config_model_dump_json():
    """Test model_dump(mode='json') produces the correct JSON stringifiable dictionary."""
    config = EngineConfig(
        chunker_id="semantic",
        vector_store="faiss_persistent",
        embedding_dim=256,
        my_custom_field="custom_value"
    )
    dumped_config = config.model_dump(mode='json')
    assert dumped_config["chunker_id"] == "semantic"
    assert dumped_config["vector_store"] == "faiss_persistent"
    assert dumped_config["embedding_dim"] == 256
    assert dumped_config["my_custom_field"] == "custom_value"

def test_engine_config_validate_assignment():
    """Test validate_assignment by assigning an invalid type to a field."""
    config = EngineConfig()
    with pytest.raises(ValidationError):
        config.chunker_id = 123
    with pytest.raises(ValidationError):
        config.vector_store = True
    with pytest.raises(ValidationError):
        config.embedding_dim = []

    # Check that valid assignment still works
    config.embedding_dim = 512
    assert config.embedding_dim == 512
    config.chunker_id = "another_chunker"
    assert config.chunker_id == "another_chunker"
