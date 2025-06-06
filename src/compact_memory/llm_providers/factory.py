import os
from typing import Optional, Any, Dict
from compact_memory.config import Config
from compact_memory.llm_providers_abc import LLMProvider
from compact_memory.llm_providers import OpenAIProvider, MockLLMProvider
# Optional providers will be imported within create_llm_provider with try-except

def create_llm_provider(
    config_name: Optional[str] = None,
    provider_type: Optional[str] = None,
    model_name: Optional[str] = None, # Primarily for context, not direct use in factory for provider instantiation
    api_key: Optional[str] = None,
    app_config: Optional[Config] = None,
) -> LLMProvider:
    """
    Creates an LLM provider instance based on configuration name or direct parameters.
    """
    if not app_config:
        app_config = Config()

    llm_config_data: Optional[Dict[str, Any]] = None
    # model_path: Optional[str] = None # Not directly used by factory for now

    if config_name:
        llm_config_data = app_config.get_llm_config(config_name)
        if not llm_config_data:
            available_configs = list(app_config.get_all_llm_configs().keys())
            raise ValueError(
                f"LLM configuration '{config_name}' not found in llm_models_config.yaml. "
                f"Available configurations: {available_configs}"
            )

        # Values from YAML config override direct CLI flags if config_name is used
        provider_type = llm_config_data.get("provider", provider_type)
        api_key = llm_config_data.get("api_key", api_key)
        # model_path = llm_config_data.get("model_path") # For future use if providers take path at init

    if not provider_type:
        # This should ideally be caught by CLI validation (e.g. if --llm-config is invalid and no --llm-provider-type)
        raise ValueError("LLM provider type must be specified either via a valid --llm-config or --llm-provider-type.")

    provider_instance: Optional[LLMProvider] = None
    provider_type_lower = provider_type.lower()

    if provider_type_lower == "openai":
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key # OpenAI client typically checks this env var
        provider_instance = OpenAIProvider()
    elif provider_type_lower == "mock":
        provider_instance = MockLLMProvider()
    elif provider_type_lower == "local":
        try:
            from compact_memory.llm_providers import LocalTransformersProvider
            if LocalTransformersProvider is None: # Check if it was set to None in __init__.py due to import error
                raise ImportError("LocalTransformersProvider is None, likely due to missing dependencies.")
            provider_instance = LocalTransformersProvider()
        except ImportError as e:
            raise ImportError(
                "Local LLM provider (LocalTransformersProvider) could not be imported. "
                "Ensure 'transformers' (and 'torch') are installed (e.g., 'pip install compact-memory[local]'). "
                f"Original error: {e}"
            )
    elif provider_type_lower == "gemini":
        try:
            from compact_memory.llm_providers import GeminiProvider
            if GeminiProvider is None: # Check if it was set to None in __init__.py
                raise ImportError("GeminiProvider is None, likely due to missing dependencies.")
            if api_key: # Gemini client uses GOOGLE_API_KEY
                os.environ["GOOGLE_API_KEY"] = api_key
            provider_instance = GeminiProvider()
        except ImportError as e:
            raise ImportError(
                "Gemini LLM provider (GeminiProvider) could not be imported. "
                "Ensure 'google-generativeai' is installed (e.g., 'pip install compact-memory[gemini]'). "
                f"Original error: {e}"
            )
    else:
        # Consider dynamically listing available provider types from __init__.py or a registry
        known_provider_slugs = ["openai", "local", "mock", "gemini"]
        raise ValueError(
            f"Unsupported LLM provider type: '{provider_type}'. "
            f"Supported types (if dependencies are installed): {', '.join(known_provider_slugs)}."
        )

    if provider_instance is None:
         # This path should ideally not be reached if the logic above is correct and covers all cases.
         raise RuntimeError(f"Internal error: Could not create an LLM provider for type '{provider_type}'.")

    return provider_instance
