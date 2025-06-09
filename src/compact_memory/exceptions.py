"""
Custom exceptions for the Compact Memory library.
"""

class CompactMemoryError(Exception):
    """Base class for all Compact Memory library specific errors."""
    pass

class EngineError(CompactMemoryError):
    """Base class for errors related to compression engines."""
    pass

class EngineLoadError(EngineError):
    """Raised when an engine fails to load from disk."""
    pass

class EngineSaveError(EngineError):
    """Raised when an engine fails to save to disk."""
    pass

class VectorStoreError(CompactMemoryError):
    """Base class for errors related to vector stores."""
    pass

class IndexRebuildError(VectorStoreError):
    """Raised when rebuilding a vector store's index fails."""
    pass

class ConfigurationError(CompactMemoryError):
    """Raised for general configuration issues."""
    pass

class EmbeddingDimensionMismatchError(EngineError): # Or ConfigurationError, EngineError seems suitable
    """
    Raised when there's a mismatch in embedding dimensions.
    For example, when loading an engine with a vector store that has a
    different embedding dimension than expected or configured.
    """
    pass
