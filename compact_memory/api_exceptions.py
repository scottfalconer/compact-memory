from typing import Dict, Optional, Any

class CompactMemoryError(Exception):
    """
    Base class for all custom exceptions raised by the Compact Memory API.

    Attributes:
        message (str): A human-readable description of the error.
        details (Optional[Dict[str, Any]]): A dictionary containing additional
                                             context-specific information about the error.
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details if details is not None else {}

    def __str__(self):
        if self.details:
            return f"{super().__str__()} (Details: {self.details})"
        return super().__str__()

class ConfigurationError(CompactMemoryError):
    """
    Raised when there is an error in the provided API, agent, or component configuration.

    This could include invalid values, missing required settings, or incompatible
    combinations of parameters. The 'details' attribute may contain information
    about the specific configuration key or value that caused the error.
    """
    pass

class InitializationError(CompactMemoryError):
    """
    Raised when a component of the Compact Memory system fails to initialize correctly.

    This can occur if a required resource is unavailable, a configuration prerequisite
    is not met, or an underlying library fails during its setup. The 'details'
    attribute may specify the component or resource that failed.
    """
    pass

class StrategyNotFoundError(CompactMemoryError):
    """
    Raised when a specified compression strategy cannot be found or loaded.

    This typically occurs if the strategy ID provided in a configuration or method call
    does not correspond to any registered compression strategy class. The 'details'
    may include the attempted strategy ID.
    """
    pass

class CompressionError(CompactMemoryError):
    """
    Raised if an error occurs during the text compression process within a strategy.

    This is a general error for issues encountered while a strategy is actively
    processing and compressing memories. The 'details' attribute might contain
    strategy-specific error information.
    """
    pass

class IngestionError(CompactMemoryError):
    """
    Raised when data ingestion into the memory store fails.

    This can be due to issues with data formatting, embedding generation failures
    for the input data, or problems with the underlying storage mechanism.
    The 'details' may provide context about the item(s) that failed ingestion.
    """
    pass

class RetrievalError(CompactMemoryError):
    """
    Raised when data retrieval from the memory store fails.

    This could be due to issues querying the store, problems with query embeddings,
    or if the requested data cannot be found or accessed. The 'details' may
    contain information about the query or access parameters.
    """
    pass

class LLMProviderError(CompactMemoryError):
    """
    Raised for errors related to interactions with a Large Language Model (LLM) provider.

    This includes issues such as API connection errors, authentication failures,
    rate limits, errors returned by the LLM API for a given prompt, or
    misconfiguration of the LLM provider settings. The 'details' may include
    provider-specific error codes or messages.
    """
    pass

class TokenizerError(CompactMemoryError):
    """
    Raised for errors related to text tokenization.

    This can occur if a specified tokenizer cannot be loaded, if there are issues
    during the encoding or decoding process, or if the tokenizer is not available
    when required by a component. The 'details' may include the tokenizer name
    or the text that caused the error.
    """
    pass

class BudgetExceededError(CompactMemoryError):
    """
    Raised when a requested operation would exceed a predefined budget.

    This is commonly used when an operation (e.g., text compression, LLM interaction)
    is constrained by a token limit, and the input or output would surpass this limit.
    The 'details' may include information about the budget type (e.g., 'tokens'),
    the limit, and the attempted usage.
    """
    pass

# Example of how to raise one of these:
# raise ConfigurationError("Invalid API key for OpenAI", details={"provider": "OpenAI", "key_hint": "****key"})
