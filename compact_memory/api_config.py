from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import yaml
from pydantic import BaseModel, Field

class EmbeddingConfig(BaseModel):
    """Configuration for embedding models used to generate vector representations of text."""
    provider: str = Field(..., description="The provider for the embedding model. Examples: 'huggingface', 'openai', 'cohere'.")
    model_name: str = Field(..., description="The specific name or path of the embedding model. For HuggingFace, this could be 'sentence-transformers/all-MiniLM-L6-v2'.")
    api_key: Optional[str] = Field(None, description="API key, if required by the embedding provider (e.g., OpenAI, Cohere).")
    options: Optional[Dict[str, Any]] = Field(None, description="Additional provider-specific options, e.g., region, endpoint, batch_size.")

class ChunkerConfig(BaseModel):
    """Configuration for text chunking strategies, defining how text is split into smaller pieces."""
    type: str = Field(..., description="The type of chunker to use. Examples: 'fixed_size', 'sentence_window', 'semantic'.")
    params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Parameters specific to the chosen chunker type, e.g., {'chunk_size': 512, 'overlap': 50} for 'fixed_size'.")

class LLMProviderAPIConfig(BaseModel):
    """Configuration for Large Language Model (LLM) providers used for text generation or other LLM-based tasks."""
    provider: str = Field(..., description="The provider for the LLM. Examples: 'local', 'openai', 'gemini', 'anthropic'.")
    model_name: str = Field(..., description="The specific name or path of the LLM. For local models, this might be a HuggingFace model ID.")
    api_key: Optional[str] = Field(None, description="API key, if required by the LLM provider.")
    generation_kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Default keyword arguments for LLM generation, e.g., {'temperature': 0.7, 'max_tokens': 150}.")

class StrategyConfig(BaseModel):
    """Configuration for a single compression strategy instance within the Compact Memory agent."""
    id: str = Field(..., description="A unique identifier for this strategy instance, e.g., 'gist_summary_fast'.")
    strategy_class_id: str = Field(..., description="The registered class ID of the compression strategy to use, e.g., 'SummarizerStrategy', 'FirstLastStrategy'.")
    params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Parameters specific to this strategy instance, e.g., {'max_tokens': 100} for a summarization strategy.")
    llm_config: Optional[LLMProviderAPIConfig] = Field(None, description="Optional LLM configuration specific to this strategy. If provided, it overrides the agent's default LLM for this strategy.")
    tokenizer_name: Optional[str] = Field(None, description="Optional tokenizer name specific to this strategy (e.g., 'gpt2'). If provided, it overrides the agent's default tokenizer for this strategy.")

class MemoryStoreConfig(BaseModel):
    """Configuration for the memory store where text chunks and their embeddings are stored and retrieved."""
    type: str = Field(..., description="The type of memory store to use. Examples: 'default_json_npy', 'chroma', 'weaviate', 'pinecone'.")
    path: Optional[str] = Field(None, description="Filesystem path for file-based stores like 'default_json_npy'. Not used for cloud-based stores.")
    params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Parameters specific to the chosen memory store type, e.g., connection details for cloud stores, or specific settings for local stores.")

class CompactMemoryConfig(BaseModel):
    """
    Main configuration object for the Compact Memory Agent.
    This model aggregates configurations for all components of the agent,
    including default settings and specific strategy configurations.
    """
    version: str = Field("1.0", description="Version of the configuration schema.")
    default_embedding_config: EmbeddingConfig = Field(..., description="Default configuration for the embedding pipeline used by the agent.")
    default_chunker_config: ChunkerConfig = Field(..., description="Default configuration for the text chunker used by the agent.")
    default_llm_provider_config: Optional[LLMProviderAPIConfig] = Field(None, description="Default LLM provider configuration for the agent. Can be overridden by individual strategies.")
    default_tokenizer_name: Optional[str] = Field("gpt2", description="Default tokenizer name (e.g., from HuggingFace) used by the agent, primarily for strategies that don't specify their own. Example: 'gpt2', 'claude-tokenizer'.")
    strategies: Dict[str, StrategyConfig] = Field(..., description="A dictionary where keys are unique strategy instance names (e.g., 'SummaryFast', 'QARetrieval') and values are their specific StrategyConfig objects.")
    memory_store_config: MemoryStoreConfig = Field(..., description="Configuration for the memory store used by the agent.")

    def save_to_file(self, filepath: Union[str, Path]):
        """
        Saves the current configuration to a YAML file.

        Args:
            filepath: The path (string or Path object) to the YAML file where the configuration will be saved.
                      Parent directories will be created if they do not exist.
        """
        file_path_obj = Path(filepath)
        file_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path_obj, 'w') as f:
            yaml.dump(self.model_dump(), f, indent=2)

    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> 'CompactMemoryConfig':
        """
        Loads a CompactMemoryConfig from a YAML file.

        Args:
            filepath: The path (string or Path object) to the YAML file from which to load the configuration.

        Returns:
            An instance of CompactMemoryConfig parsed from the YAML file.

        Raises:
            FileNotFoundError: If the specified filepath does not exist.
            yaml.YAMLError: If the file content is not valid YAML.
            pydantic.ValidationError: If the loaded data does not conform to the CompactMemoryConfig schema.
        """
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
