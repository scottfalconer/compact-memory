# Developing Compression Engines

This guide provides a comprehensive walkthrough for researchers and developers looking to create new `BaseCompressionEngine` implementations within the Compact Memory framework. It covers the core concepts, practical steps, and best practices for building, testing, and integrating your custom engines.

## Core Concept: The `BaseCompressionEngine`

At the heart of Compact Memory's extensibility is the `BaseCompressionEngine` abstract base class. Any new engine you develop must inherit from this class and implement its required methods.

### Abstract Base Class: `compact_memory.engines.BaseCompressionEngine`

```python
from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Any, Optional, Dict

from compact_memory.engines import CompressionTrace, CompressedMemory

class BaseCompressionEngine(ABC):
    # Unique identifier for your engine. This is crucial for registration and selection.
    id: str

    @abstractmethod
    def compress(
        self,
        text: str, # Standardized to 'text'
        budget: int, # Standardized to 'budget'
        previous_compression_result: Optional[CompressedMemory] = None, # Added
        **kwargs: Any,
    ) -> CompressedMemory: # Changed return type
        """
        Compresses the input text to meet the token budget.

        Args:
            text: The input string to compress.
            budget: The target maximum number of tokens (or a proxy) for the compressed output.
            previous_compression_result (Optional[CompressedMemory]): Output from a preceding
                                                                    engine in a pipeline.
            **kwargs: Additional keyword arguments, commonly including:
                - `tokenizer`: Optional tokenizer for accurate token counting.
                - `source_document_id` (Optional): Identifier for the source.
                - Other engine-specific parameters.

        Returns:
            A CompressedMemory object containing the compressed text and related information
            (engine_id, engine_config, trace, metadata).
        """
        pass

    # Optional methods for engines with learnable components
    def save_learnable_components(self, path: str) -> None:
        """Persist any trainable state to `path`."""
        # pragma: no cover - optional
        pass

    def load_learnable_components(self, path: str) -> None:
        """Load previously saved trainable state from `path`."""
        # pragma: no cover - optional
        pass
```

### Implementing `compress()`

Your primary task is to implement the `compress` method. Here's what to consider:

1.  **Input (`text_or_chunks`):**
    *   Decide if your engine works best with a single block of text or pre-chunked text.
    *   If you expect chunks, you might need to join them or process them individually.
    *   If you receive a single string, you might need to implement chunking logic within your engine or use a provided chunker.

2.  **Token Budget (`llm_token_budget`):**
    *   This is a crucial constraint. Your engine must try to produce output that, when tokenized, is close to this budget.
    *   If a `tokenizer` is provided in `**kwargs`, use it for accurate counting.
    *   If no `tokenizer` is available, you might fall back to character counts or word counts as a proxy, but document this limitation.
    *   Consider edge cases: What if the budget is too small for any meaningful output?

3.  **Compression Logic:**
    *   This is where your novel algorithm resides (e.g., extractive summarization, abstractive summarization, selective pruning, concept extraction, etc.).

4.  **Output (`CompressedMemory`):**
    *   The `text: str` field holds the compressed string.
    *   The `engine_id: Optional[str]` field should store `self.id`.
    *   The `engine_config: Optional[Dict[str, Any]]` field should store `self.config` or relevant parameters.
    *   The `metadata: Optional[Dict[str, Any]]` field can store any other useful information.

5.  **Tracing (`CompressionTrace` to `CompressedMemory.trace`):**
    *   The `CompressionTrace` object, previously returned separately, must now be instantiated and assigned to the `trace` field of the `CompressedMemory` object you return.
    *   Populate `CompressionTrace` with `engine_name=self.id`, relevant `strategy_params` (like `budget` and other kwargs used), `input_summary`, `output_summary`, `steps`, `processing_ms`, and `final_compressed_object_preview`.
    *   Refer to `docs/EXPLAINABLE_COMPRESSION.md` for standard vocabulary for trace step types.

6.  **Handling `previous_compression_result`:**
    *   If `previous_compression_result` is provided, your engine can use its `text`, `metadata`, `trace`, `engine_id`, or `engine_config` to inform its own compression process.
    *   For example, it might operate on `previous_compression_result.text` instead of the `text` argument if chaining is intended this way for your engine.
    *   If your engine doesn't use it, it can be ignored.

### Example: A Simple Truncation Engine

```python
from compact_memory.engines import BaseCompressionEngine, CompressedMemory, CompressionTrace
from compact_memory.token_utils import get_tokenizer, token_count

class SimpleTruncateEngine(BaseCompressionEngine):
    id = "simple_truncate"

    def compress(
        self,
        text: str,
        budget: int,
        previous_compression_result: Optional[CompressedMemory] = None,
        **kwargs
    ) -> CompressedMemory:

        # Determine actual text to process
        input_text = text
        if previous_compression_result:
            # Example: This engine could choose to always process the original text,
            # or use previous_compression_result.text. For simplicity, let's use 'text'.
            # For a chaining behavior, one might do:
            # input_text = previous_compression_result.text
            pass


        tokenizer = kwargs.get("tokenizer")
        # Fallback to character-based if no tokenizer
        actual_tokenizer_for_count = tokenizer if tokenizer else lambda x: list(x)

        original_length_chars = len(input_text)
        original_tokens = token_count(actual_tokenizer_for_count, input_text)

        # Simple truncation logic
        # A real engine would use the tokenizer to count and truncate tokens.
        # This example uses character limit as a proxy if no good tokenizer.
        limit = budget
        if not tokenizer or tokenizer == list: # If using char-based fallback for token counting
             # Assuming average 4 chars per token if no real tokenizer for budget
            limit = budget * 4 # Character limit

        compressed_text = input_text[:limit]

        # Refine if a real tokenizer was provided and budget is exceeded
        if tokenizer and tokenizer != list:
            current_tokens = token_count(tokenizer, compressed_text)
            while current_tokens > budget and len(compressed_text) > 0:
                # Naively remove characters/words until budget is met
                compressed_text = compressed_text[:-10] if len(compressed_text) > 10 else ""
                current_tokens = token_count(tokenizer, compressed_text)
            if current_tokens > budget: # Final hard truncate by token IDs
                 encoded_ids = tokenizer(compressed_text)['input_ids'][:budget]
                 if hasattr(tokenizer, "decode"):
                     compressed_text = tokenizer.decode(encoded_ids)
                 else: # Basic fallback if no decode
                     # This part is tricky without a full tokenizer interface,
                     # actual tokenizers handle this better.
                     compressed_text = f"Could not decode {len(encoded_ids)} tokens"


        final_compressed_tokens = token_count(actual_tokenizer_for_count, compressed_text)
        final_compressed_chars = len(compressed_text)

        current_trace = CompressionTrace(
            engine_name=self.id,
            strategy_params={"budget": budget, "method": "truncation"},
            input_summary={"original_length_chars": original_length_chars, "original_tokens": original_tokens},
            steps=[
                {"type": "truncation_attempt", "limit_type": "chars" if not tokenizer or tokenizer == list else "tokens", "limit_value": budget}
            ],
            output_summary={"compressed_length_chars": final_compressed_chars, "compressed_tokens": final_compressed_tokens},
            final_compressed_object_preview=compressed_text[:50]
        )

        return CompressedMemory(
            text=compressed_text,
            engine_id=self.id,
            engine_config=self.config, # Assuming self.config is set in __init__
            trace=current_trace,
            metadata={"truncation_details": "simple character or token based"}
        )
```

## Handling Token Budgets and Tokenizers

Effective budget management is key.

*   **Prioritize `tokenizer`:** If `kwargs['tokenizer']` is available, use it. This allows for precise token counting and manipulation. Compact Memory often uses `tiktoken` (e.g., `get_tokenizer("gpt2")`) or tokenizers from the `transformers` library.
*   **Fallback Mechanisms:** If no tokenizer is provided, your engine must have a fallback. This could be:
    *   Character counts (e.g., assuming an average of 3-4 characters per token).
    *   Word counts.
    *   Clearly document this assumption and its potential inaccuracies.
*   **Iterative Refinement:** Some engines might need to iteratively refine the output to meet the budget, especially after summarization or transformation steps that can change token counts unpredictably.
*   **Over-budget Handling:** Decide how to handle cases where even minimal content exceeds the budget. Return an empty string? A specific warning in the trace?

## Accessing Shared Utilities

Compact Memory provides utilities that can be helpful:

*   **Tokenizers:**
    *   `compact_memory.token_utils.get_tokenizer(tokenizer_name_or_path)`: Helper to load `tiktoken` or `transformers` tokenizers.
    *   `compact_memory.token_utils.token_count(tokenizer, text)`: Counts tokens in a text using the provided tokenizer.
*   **Chunking:**
    *   While engines can implement their own chunking, Compact Memory also has chunking utilities (e.g., `SentenceWindowChunker`) that can be used externally to prepare input for your engine or internally if your engine requires chunk-based processing. See `compact_memory.chunker`.
*   **LLM Helpers (Optional):**
    *   If your engine needs to call an LLM, Compact Memory keeps this outside the core package. Check `examples/llm_helpers.py` for lightweight `run_llm()` wrappers that work with small local models or OpenAI.
    *   You can use these helpers directly or swap in your preferred framework (LangChain, AutoGen, etc.). The helpers simply take a prompt and return the generated text.
    *   Remember to manage API keys and errors in your own code when using external providers.

## Structuring Engine Logic

*   **Modularity:** Keep your compression logic well-organized. Helper methods for distinct steps (e.g., preprocessing, core compression, postprocessing) can improve readability.
*   **Configuration:** If your engine has tunable parameters (e.g., summarization model, number of sentences to keep), make them arguments to `__init__` with sensible defaults. These parameters should be recorded in the `CompressionTrace`.
*   **State:**
    *   Most engines should aim to be stateless within the `compress` call for a given input.
    *   If your engine has *learnable components* (e.g., a fine-tuned model), implement `save_learnable_components` and `load_learnable_components` to manage its state across sessions.

## Testing Your Engine

Rigorous testing is crucial when developing new engines.

1.  **Unit Tests:**
    *   Write standard Python unit tests for your engine's core logic. Test edge cases, different input types, and budget handling.
    *   Mock external dependencies like LLM calls if necessary.


## Registering Your Engine

For Compact Memory to find and use your engine, it needs to be registered.

*   **Plugin System:** The preferred way is through the plugin system. If your engine is part of an installable Python package, you can register it via an entry point in your `pyproject.toml` or `setup.py`. See `docs/SHARING_ENGINES.md`.
*   **Direct Registration (for local development/testing):**
    ```python
    from compact_memory.registry import register_compression_engine
    from .my_engine_module import MyCustomEngine

    register_compression_engine(MyCustomEngine.id, MyCustomEngine)
    ```
    This is useful in scripts or during development before packaging.

## Best Practices

*   **Clarity and Simplicity:** Aim for understandable code.
*   **Efficiency:** Be mindful of computational cost, especially if your engine is complex or calls external services.
*   **Robustness:** Handle potential errors gracefully (e.g., invalid inputs, API failures).
*   **Comprehensive Tracing:** Good traces are invaluable for users and for your own debugging.
*   **Documentation:**
    *   Add detailed docstrings to your engine class and methods.
    *   If your engine has unique dependencies or setup requirements, document them in a `README.md` if you package it.
*   **Distribution:** When publishing on PyPI or GitHub, use the package naming pattern `compact_memory_<name>_engine`.

By following this guide, you can effectively contribute new and innovative compression engines to the Compact Memory ecosystem.
