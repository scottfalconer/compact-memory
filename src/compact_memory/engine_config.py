from typing import Optional, Any, Dict
from pydantic import BaseModel, Field


class EngineConfig(BaseModel):
    """
    Configuration for a BaseCompressionEngine, including chunker, vector store,
    embedding settings, and paths for serializable custom functions.
    In Pydantic V2, extra fields are automatically
    stored in `model_extra` when `Config.extra = 'allow'`.
    """

    chunker_id: str = Field(
        default="fixed_size", description="Identifier for the chunker to use."
    )
    vector_store: str = Field(
        default="faiss_memory", description="Identifier for the vector store to use."
    )
    embedding_dim: Optional[int] = Field(
        default=None,
        description="Dimension of embeddings. If None, it might be inferred.",
    )
    vector_store_path: Optional[str] = Field(
        default=None, description="Path for persistent vector stores."
    )
    embedding_fn_path: Optional[str] = Field(
        default=None,
        description="Path to the embedding function, e.g., 'module.submodule.function_name'.",
    )
    preprocess_fn_path: Optional[str] = Field(
        default=None,
        description="Path to the preprocessing function, e.g., 'module.submodule.function_name'.",
    )
    enable_trace: bool = Field(
        default=True,
        description="Whether to generate a compression trace during the compress operation.",
    )

    # No explicit model_extra field or validator needed for Pydantic V2.
    # Extras are handled by `model.model_extra` property if `Config.extra = 'allow'`.

    # Pydantic model configuration
    # For Pydantic v2, this should be ConfigDict = {}
    # However, the warning mentioned "class-based `config` is deprecated, use ConfigDict instead."
    # For now, sticking to class Config as per existing structure, will address warning later if needed.
    class Config:
        validate_assignment = True
        extra = "allow"

    def get(self, key: str, default: Any | None = None) -> Any:
        """Retrieve ``key`` similar to ``dict.get``."""
        return self.model_extra.get(key, getattr(self, key, default))
