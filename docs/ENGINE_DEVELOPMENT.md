# Developing Compression Engines

This guide provides a comprehensive walkthrough for researchers and developers looking to create new `BaseCompressionEngine` implementations within the Compact Memory framework. It covers the core concepts, practical steps, and best practices for building, testing, and integrating your custom engines.

## Core Concept: The `BaseCompressionEngine`

At the heart of Compact Memory's extensibility is the `BaseCompressionEngine` abstract base class. Any new engine you develop must inherit from this class and implement its required methods.

### Abstract Base Class: `BaseCompressionEngine.core.engines_abc.BaseCompressionEngine`

```python
from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Any, Optional, Dict

from BaseCompressionEngine.core.trace import CompressionTrace
from BaseCompressionEngine.core.engines_abc import CompressedMemory

class BaseCompressionEngine(ABC):
    # Unique identifier for your engine. This is crucial for registration and selection.
    id: str

    @abstractmethod
    def compress(
        self,
        text_or_chunks: Union[str, List[str]],
        llm_token_budget: int,
        **kwargs: Any,
    ) -> Tuple[CompressedMemory, CompressionTrace]:
        """
        Compresses the input text or chunks to meet the token budget.

        Args:
            text_or_chunks: Either a single string of text or a list of pre-chunked strings.
                            Your engine needs to handle both cases or define its expected input.
            llm_token_budget: The target maximum number of tokens (or a proxy like characters,
                            depending on your engine's design) that the compressed output
                            should ideally have.
            **kwargs: A dictionary for additional keyword arguments. This commonly includes:
                - `tokenizer`: An optional tokenizer function (e.g., from `tiktoken` or
                  `transformers`) that can be used for accurate token counting or
                  token-aware processing. Engines should be robust to its absence.
                - `source_document_id` (Optional): An identifier for the source document, useful for context.
                - Other engine-specific parameters passed during engine instantiation or invocation.

        Returns:
            A tuple containing:
                - CompressedMemory: An object with a `text` attribute holding the compressed
                  string, and an optional `metadata` dictionary.
                - CompressionTrace: An object detailing the steps, decisions, and outcomes
                  of the compression process. This is vital for debugging and analysis.
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
    *   The `text` attribute should contain the final compressed string.
    *   The `metadata` attribute can store any useful information about the compression, like the original token count, compression ratio, or parameters used.

5.  **Tracing (`CompressionTrace`):**
    *   This is essential for transparency and debugging. Record key decisions and transformations.
    *   Instantiate `CompressionTrace` with `engine_name=self.id`, `engine_params` (any parameters your engine was initialized with or received), `input_summary` (e.g., original length/tokens), and `output_summary` (e.g., final length/tokens).
    *   Append dictionaries to `trace.steps` for each significant operation (e.g., `{"type": "chunking", "num_chunks": 5}` or `{"type": "summarization_model_call", "model_name": "t5-small"}`).
    *   Populate `trace.processing_ms` with the time taken.
    *   A `final_compressed_object_preview` is also useful.
    *   Refer to `docs/EXPLAINABLE_COMPRESSION.md` for standard vocabulary for trace step types.

### Example: A Simple Truncation Engine

```python
from BaseCompressionEngine.core.engines_abc import BaseCompressionEngine, CompressedMemory
from BaseCompressionEngine.core.trace import CompressionTrace
from compact_memory.token_utils import get_tokenizer, token_count

class SimpleTruncateEngine(BaseCompressionEngine):
    id = "simple_truncate"

    def compress(self, text_or_chunks, llm_token_budget, **kwargs):
        tokenizer = kwargs.get("tokenizer")
        if tokenizer is None:
            # Fallback: treat budget as character count if no tokenizer
            tokenizer = lambda x: list(x) # Simple char tokenizer for length
            actual_tokenizer_for_count = lambda x: list(x)
        else:
            actual_tokenizer_for_count = tokenizer

        if isinstance(text_or_chunks, list):
            input_text = " ".join(text_or_chunks)
        else:
            input_text = str(text_or_chunks)

        original_tokens = token_count(actual_tokenizer_for_count, input_text)

        # Simple truncation logic (very naive)
        # A real engine would be more sophisticated, using the tokenizer
        # to truncate based on actual tokens.
        limit = llm_token_budget
        if tokenizer is str.split or actual_tokenizer_for_count == list(input_text): # if using fallback tokenizer
             # Assuming average 4 chars per token if no real tokenizer for budget
            limit = llm_token_budget * 4

        compressed_text = input_text[:limit]

        # Refine if over budget with actual tokenizer
        if tokenizer != list(input_text): # If a real tokenizer was provided
            current_tokens = token_count(tokenizer, compressed_text)
            while current_tokens > llm_token_budget and len(compressed_text) > 0:
                # Naively remove characters/words until budget is met
                # A better approach would remove whole tokens
                compressed_text = compressed_text[:-10] if len(compressed_text) > 10 else ""
                current_tokens = token_count(tokenizer, compressed_text)
            # Final check, hard truncate if still over (e.g. one very long token)
            if current_tokens > llm_token_budget:
                 encoded = tokenizer(compressed_text)[:llm_token_budget]
                 compressed_text = tokenizer.decode(encoded) if hasattr(tokenizer, "decode") else "".join(encoded)


        final_compressed_tokens = token_count(actual_tokenizer_for_count, compressed_text)

        trace = CompressionTrace(
            engine_name=self.id,
            engine_params={"llm_token_budget": llm_token_budget},
            input_summary={"original_text_length": len(input_text), "original_tokens": original_tokens},
            steps=[
                {"type": "input_processing", "input_type": type(text_or_chunks).__name__},
                {"type": "truncation", "budget_type": "chars" if actual_tokenizer_for_count == list(input_text) else "tokens", "limit": llm_token_budget}
            ],
            output_summary={"compressed_text_length": len(compressed_text), "compressed_tokens": final_compressed_tokens},
            final_compressed_object_preview=compressed_text[:50]
        )

        return CompressedMemory(text=compressed_text), trace
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

Rigorous testing is crucial. Compact Memory's experimentation framework helps with this.

1.  **Unit Tests:**
    *   Write standard Python unit tests for your engine's core logic. Test edge cases, different input types, and budget handling.
    *   Mock external dependencies like LLM calls if necessary.

2.  **Experimentation Framework:**
    *   Compact Memory provides tools to run experiments comparing engines. This typically involves:
        *   A dataset (e.g., a collection of text files).
        *   One or more compression engines to test.
        *   Configuration for each engine (parameters, token budgets).
        *   Validation metrics to evaluate the output.
    *   **Key components:**
        *   `ExperimentConfig`, `ResponseExperimentConfig`, `HistoryExperimentConfig`: Dataclasses for defining experiment parameters.
        *   `run_experiment`, `run_response_experiment`, `run_history_experiment`: Functions to execute these experiments.
        *   `ValidationMetric`: Base class for metrics that evaluate compression quality or task performance (e.g., ROUGE scores, LLM-based evaluation). See `docs/DEVELOPING_VALIDATION_METRICS.md`.
    *   **Workflow:**
        1.  Define an experiment configuration file (often YAML) or create config objects programmatically.
        2.  Specify your engine's ID and any parameters in the config.
        3.  Run the experiment using the CLI (`compact-memory experiment run ...`) or Python API.
        4.  Analyze the output metrics to see how your engine performs.

3.  **`onboarding_demo.py`:**
    *   The `examples/onboarding_demo.py` script shows a basic example of defining a engine, registering it, and using it in an experiment. Use it as a reference.

4.  **Packaging for Experiments:**
    *   If you package your engine (see `docs/SHARING_ENGINES.md`), you can include example experiment configurations within your package. The CLI command `compact-memory dev run-package-experiment` can then execute these.

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
