"""LLM provider implementations.

The optional providers are imported lazily so missing heavy dependencies do not
break basic functionality or tests that don't require them.
"""
from typing import Optional

from compact_memory.config import Config
from compact_memory.llm_providers_abc import LLMProvider
from .openai_provider import OpenAIProvider
from .mock_provider import MockLLMProvider # Added MockLLMProvider

# Conditional imports for optional providers
try:
    from .gemini_provider import GeminiProvider
except ImportError:
    GeminiProvider = None # type: ignore
try:
    from .local_provider import LocalTransformersProvider
except ImportError:
    LocalTransformersProvider = None # type: ignore

__all__ = [
    "OpenAIProvider",
    "MockLLMProvider", # Added MockLLMProvider
    "get_llm_provider",
]

try:  # Gemini provider requires ``google-generativeai``
    # from .gemini_provider import GeminiProvider # Already imported above
    pass
except Exception:  # pragma: no cover - optional dependency may be missing
    GeminiProvider = None  # type: ignore
else:  # pragma: no cover - imported when dependency is available
    if "GeminiProvider" not in __all__ and GeminiProvider is not None:
        __all__.append("GeminiProvider")

try:  # Local provider requires transformers
    # from .local_provider import LocalTransformersProvider # Already imported above
    pass
except Exception:  # pragma: no cover - optional dependency may be missing
    LocalTransformersProvider = None  # type: ignore
else:  # pragma: no cover - imported when dependency is available
    if "LocalTransformersProvider" not in __all__ and LocalTransformersProvider is not None:
        __all__.append("LocalTransformersProvider")

def get_llm_provider(model_id: str, global_config: Config) -> Optional[LLMProvider]:
    llm_specific_config = global_config.get_llm_config(model_id)

    if not llm_specific_config:
        # Consider logging if verbose or a more formal logging system is in place
        # print(f"Debug: Configuration for model_id '{model_id}' not found in llm_models_config.yaml.")
        return None

    provider_name = llm_specific_config.get("provider")
    # The actual model_name (e.g., 'gpt-3.5-turbo', 'sshleifer/tiny-gpt2')
    # is stored in llm_specific_config.get("model_name").
    # This specific model_name is typically passed by the consuming code (e.g., an engine)
    # to the provider's methods (like generate_response(model_name=...)).
    # The factory generally just needs to return the correct provider type.

    if provider_name == "openai":
        # OpenAIProvider reads API key from environment OPENAI_API_KEY
        return OpenAIProvider()
    elif provider_name == "local":
        if LocalTransformersProvider:
            return LocalTransformersProvider()
        else:
            # print("Debug: LocalTransformersProvider not available (likely missing dependencies).")
            return None
    elif provider_name == "gemini":
        if GeminiProvider:
            # GeminiProvider reads API key from environment GEMINI_API_KEY
            return GeminiProvider()
        else:
            # print("Debug: GeminiProvider not available (likely missing dependencies).")
            return None
    elif provider_name == "mock": # Added for completeness, if MockLLMProvider is to be factory instantiable
        return MockLLMProvider()
    # Add other providers here as elif blocks
    else:
        # print(f"Debug: Unknown or unsupported provider specified for model_id '{model_id}': {provider_name}")
        return None
