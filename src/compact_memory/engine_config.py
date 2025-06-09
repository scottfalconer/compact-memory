from typing import Optional, Any, Dict
from pydantic import BaseModel, Field

class EngineConfig(BaseModel):
    """
    Configuration for a BaseCompressionEngine, including chunker, vector store,
    and embedding settings. In Pydantic V2, extra fields are automatically
    stored in `model_extra` when `Config.extra = 'allow'`.
    """
    chunker_id: str = Field(default="fixed_size", description="Identifier for the chunker to use.")
    vector_store: str = Field(default="in_memory", description="Identifier for the vector store to use.") # Changed default
    embedding_dim: Optional[int] = Field(default=None, description="Dimension of embeddings. If None, it might be inferred.")
    vector_store_path: Optional[str] = Field(default=None, description="Path for persistent vector stores.")

    # No explicit model_extra field or validator needed for Pydantic V2.
    # Extras are handled by `model.model_extra` property if `Config.extra = 'allow'`.

    # Pydantic model configuration
    # For Pydantic v2, this should be ConfigDict = {}
    # However, the warning mentioned "class-based `config` is deprecated, use ConfigDict instead."
    # For now, sticking to class Config as per existing structure, will address warning later if needed.
    class Config:
        validate_assignment = True
        extra = 'allow'
