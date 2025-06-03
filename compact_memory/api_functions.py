from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Any, Union # Added Union

# API Models and Exceptions
from .api_models import CompressedMemoryContext, SourceReference
from .api_exceptions import StrategyNotFoundError, CompressionError, ConfigurationError

# API Models and Exceptions (already imported or assumed)
# from .api_models import CompressedMemoryContext, SourceReference # SourceReference already imported
# from .api_exceptions import StrategyNotFoundError, CompressionError, ConfigurationError # Already imported

from .compression.strategies_abc import CompressionStrategy # Real ABC
from .registry import get_compression_strategy_class # Real registry function
from .token_utils import Tokenizer, get_tokenizer # Real tokenizer utilities
from .llm_providers_abc import LLMProvider # Real LLM Provider ABC
from .llm_providers import get_llm_provider # Real helper to get LLM provider instances
from .api_config import LLMProviderAPIConfig # To potentially create LLM provider on the fly

# Assuming internal models are accessible for mapping, or use placeholders
# from .models import CompressedMemory # This is an internal model, not API one
# from .compression.trace import CompressionTrace # This is an internal model

logger = logging.getLogger(__name__)


def compress_text(
    text: str,
    strategy_class_id: str,
    budget: int,
    strategy_params: Optional[Dict[str, Any]] = None,
    tokenizer_instance: Optional[Tokenizer] = None,
    llm_provider_instance: Optional[LLMProvider] = None
) -> CompressedMemoryContext:
    """Statelessly compresses input text using a specified strategy and budget.

    This function provides a direct way to access text compression capabilities
    without needing to manage a full CompactMemoryAgent instance. It loads,
    instantiates, and runs the specified compression strategy on the given text.

    Args:
        text: The raw string of text to be compressed.
        strategy_class_id: The registered class identifier of the compression
                           strategy to be used (e.g., "FirstLastStrategy",
                           "SummarizerStrategy").
        budget: The target budget for the compressed output. The interpretation
                of this budget (e.g., max tokens, characters) can be
                strategy-dependent, but typically refers to tokens.
        strategy_params: A dictionary of parameters to initialize the
                               chosen strategy instance. These are specific to the
                               selected strategy class. Defaults to None (no params).
        tokenizer_instance: An optional pre-initialized tokenizer instance.
                                  Some strategies may require a tokenizer for effective
                                  operation or accurate budgeting. If required by the
                                  strategy and not provided, a ConfigurationError
                                  may be raised. Defaults to None.
        llm_provider_instance: An optional pre-initialized LLM provider instance.
                                    Some strategies (e.g., learned summarizers) require
                                    an LLM. If required by the strategy and not
                                    provided, a ConfigurationError may be raised.
                                    Defaults to None.

    Returns:
        A CompressedMemoryContext object containing the compressed text,
        source references (often a single reference to the input text if the
        strategy doesn't produce more detailed ones), strategy information,
        budget details, and trace information.

    Raises:
        ValueError: If the input text is empty.
        StrategyNotFoundError: If the specified `strategy_class_id` is not found
                               in the strategy registry.
        ConfigurationError: If a strategy requires a tokenizer or LLM provider
                            and a compatible instance is not provided.
        CompressionError: If any other error occurs during the compression
                          process itself.
    """
    logger.info(
        f"Compressing text (first 50 chars): '{text[:50]}...' using strategy_class_id: {strategy_class_id}, budget: {budget}"
    )
    start_time = time.time()

    if not text:
        logger.warning("compress_text called with empty text.")
        # Return an empty context or raise error, consistent with agent methods
        raise ValueError("Input text cannot be empty for compression.")

    if budget <= 0:
        logger.warning(f"Compression budget is {budget}. Result might be empty or unconstrained.")
        # Depending on strategy, a non-positive budget might mean different things.
        # For now, we proceed but strategies should handle this.

    try:
        # 1. Load the compression strategy class
        strategy_cls = get_compression_strategy_class(strategy_class_id)
        logger.info(f"Loaded strategy class: {strategy_cls.__name__}")

        # 2. Instantiate the strategy
        # Real strategies primarily take `params` in __init__. Tokenizer/LLM are passed to `compress`.
        strategy_instance = strategy_cls(params=(strategy_params or {}))
        logger.info(f"Instantiated strategy '{strategy_instance.id}' with params: {strategy_params}")

        # Configuration Checks (after instantiation, based on strategy properties)
        if getattr(strategy_instance, 'requires_llm', False) and not llm_provider_instance:
            raise ConfigurationError(
                f"Strategy '{strategy_instance.id}' requires an LLM provider, but none was provided.",
                details={"strategy_class_id": strategy_class_id}
            )
        if getattr(strategy_instance, 'requires_tokenizer', False) and not tokenizer_instance:
            raise ConfigurationError(
                f"Strategy '{strategy_instance.id}' requires a tokenizer, but none was provided.",
                details={"strategy_class_id": strategy_class_id}
            )

        # 3. Prepare kwargs for the strategy's compress method
        compress_kwargs = {
            "tokenizer": tokenizer_instance,
            "llm_provider": llm_provider_instance,
        }

        # 4. Execute the strategy's compress method
        # Real strategies return Tuple[CompressedMemory, CompressionTrace]
        # Assuming CompressedMemory and CompressionTrace are importable or their structure is known for mapping
        compressed_memory_obj, compression_trace_obj = strategy_instance.compress(
            text_or_chunks=text, # Pass the raw text
            budget=budget,
            **compress_kwargs
        )

        # 5. Adapt the result to CompressedMemoryContext API model
        source_refs = []
        # Assuming compressed_memory_obj might have 'source_chunks' or similar if it's from summarization of multiple inputs
        # For a single text input, direct source_references might be less common from the strategy itself.
        # If the strategy populates source_references on the CompressedMemory object:
        if hasattr(compressed_memory_obj, 'source_references') and compressed_memory_obj.source_references:
            for ref in compressed_memory_obj.source_references: # Assuming internal ref format
                source_refs.append(SourceReference(
                    document_id=getattr(ref, 'document_id', None), # internal ref might be simple dict or obj
                    chunk_id=getattr(ref, 'chunk_id', None),
                    text_snippet=getattr(ref, 'text', '') if hasattr(ref, 'text') else getattr(ref, 'text_snippet', ''),
                    score=getattr(ref, 'score', None),
                    metadata=getattr(ref, 'metadata', {})
                ))
        elif not isinstance(text, list): # If input was single string and no refs from strategy
             source_refs.append(SourceReference(text_snippet=text[:200], document_id="original_input_text"))


        cm_context = CompressedMemoryContext(
            compressed_text=compressed_memory_obj.text, # Assuming .text attribute
            source_references=source_refs,
            strategy_id_used=strategy_instance.id,
            budget_info={
                "requested_budget": budget,
                "final_tokens": getattr(compression_trace_obj, 'compressed_tokens', None),
                "original_tokens": getattr(compression_trace_obj, 'original_tokens', None),
            },
            processing_time_ms=(time.time() - start_time) * 1000, # Default, may be overwritten by trace
            strategy_llm_input=getattr(compression_trace_obj, 'llm_input', None),
            strategy_llm_output=getattr(compression_trace_obj, 'llm_output', None),
            full_trace=compression_trace_obj.to_dict() if hasattr(compression_trace_obj, 'to_dict') else vars(compression_trace_obj)
        )

        if hasattr(compression_trace_obj, 'processing_ms') and getattr(compression_trace_obj, 'processing_ms') is not None:
            cm_context.processing_time_ms = compression_trace_obj.processing_ms

        logger.info(f"Text compression completed in {cm_context.processing_time_ms:.2f} ms.")
        return cm_context

    except StrategyNotFoundError as e:
        logger.error(f"Strategy not found: {e}", exc_info=True)
        raise
    except ConfigurationError as e: # Catch config errors from strategy instantiation
        logger.error(f"Configuration error for strategy: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during text compression: {e}", exc_info=True)
        raise CompressionError(f"Failed to compress text: {e}", details={"original_exception": str(e)})
