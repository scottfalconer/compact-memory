# Developing Compression Strategies

This guide provides a comprehensive walkthrough for researchers and developers looking to create new `CompressionStrategy` implementations within the Compact Memory framework. It covers the core concepts, practical steps, and best practices for building, testing, and integrating your custom strategies.

## Core Concept: The `CompressionStrategy`

At the heart of Compact Memory's extensibility is the `CompressionStrategy` abstract base class. Any new strategy you develop must inherit from this class and implement its required methods.

### Abstract Base Class: `compact_memory.compression.strategies_abc.CompressionStrategy`

```python
from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Any, Optional, Dict

from compact_memory.compression.trace import CompressionTrace
from compact_memory.compression.strategies_abc import CompressedMemory

class CompressionStrategy(ABC):
    # Unique identifier for your strategy. This is crucial for registration and selection.
    id: str

    @abstractmethod
    def compress(
        self,
        text: str, # Changed
        llm_token_budget: int,
        chunk_fn: Optional[ChunkFn] = None, # Added
        **kwargs: Any,
    ) -> Tuple[CompressedMemory, CompressionTrace]:
        """
        Compresses the input text to meet the token budget.

        Args:
            text: str: The raw input text to be compressed.
            llm_token_budget: The target maximum number of tokens (or a proxy like characters,
                            depending on your strategy's design) that the compressed output
                            should ideally have.
            chunk_fn: Optional[ChunkFn]: An optional callable (`Callable[[str], List[str]]`).
                      If provided, the strategy should use this function to split the `text`
                      into a list of chunks. If `None`, the strategy should typically
                      treat the entire `text` as a single chunk (e.g., by wrapping it as `[text]`).
            **kwargs: A dictionary for additional keyword arguments. This commonly includes:
                - `tokenizer`: An optional tokenizer function (e.g., from `tiktoken` or
                  `transformers`) that can be used for accurate token counting or
                  token-aware processing. Strategies should be robust to its absence.
                - `source_document_id` (Optional): An identifier for the source document, useful for context.
                - Other strategy-specific parameters passed during strategy instantiation or invocation.

        Returns:
            A tuple containing:
                - CompressedMemory: An object with a `text` attribute holding the compressed
                  string, and an optional `metadata` dictionary.
                - CompressionTrace: An object detailing the steps, decisions, and outcomes
                  of the compression process. This is vital for debugging and analysis.
        """
        pass

    # Optional methods for strategies with learnable components
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

1.  **Input (`text` and `chunk_fn`):**
    *   The primary input is `text: str`.
    *   An optional `chunk_fn: Callable[[str], List[str]]` is also provided.
    *   If `chunk_fn` is given, your strategy should call it to split the input `text` into a list of strings (chunks):
        `chunks = chunk_fn(text)`.
    *   If `chunk_fn` is `None`, the `text` should typically be treated as a single chunk:
        `chunks = [text]`.
    *   The rest of your strategy's logic should then operate on this `chunks` list (which will contain one or more strings).
        ```python
        # Inside your strategy's compress method:
        if chunk_fn:
            chunks = chunk_fn(text)
        else:
            chunks = [text] # Treat the whole text as a single chunk

        # Now process the 'chunks' list
        processed_text = self._process_chunks(chunks) # Example helper
        ```

2.  **Token Budget (`llm_token_budget`):**
    *   This is a crucial constraint. Your strategy must try to produce output that, when tokenized, is close to this budget.
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
    *   Instantiate `CompressionTrace` with `strategy_name=self.id`, `strategy_params` (any parameters your strategy was initialized with or received), `input_summary` (e.g., original length/tokens), and `output_summary` (e.g., final length/tokens).
    *   Append dictionaries to `trace.steps` for each significant operation (e.g., `{"type": "chunking", "num_chunks": 5}` or `{"type": "summarization_model_call", "model_name": "t5-small"}`).
    *   Populate `trace.processing_ms` with the time taken.
    *   A `final_compressed_object_preview` is also useful.
    *   Refer to `docs/EXPLAINABLE_COMPRESSION.md` for standard vocabulary for trace step types.

### Example: A Simple Truncation Strategy

```python
from compact_memory.compression.strategies_abc import CompressionStrategy, CompressedMemory
from compact_memory.compression.trace import CompressionTrace
from compact_memory.token_utils import get_tokenizer, token_count

class SimpleTruncateStrategy(CompressionStrategy):
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
        # A real strategy would be more sophisticated, using the tokenizer
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
            strategy_name=self.id,
            strategy_params={"llm_token_budget": llm_token_budget},
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
*   **Fallback Mechanisms:** If no tokenizer is provided, your strategy must have a fallback. This could be:
    *   Character counts (e.g., assuming an average of 3-4 characters per token).
    *   Word counts.
    *   Clearly document this assumption and its potential inaccuracies.
*   **Iterative Refinement:** Some strategies might need to iteratively refine the output to meet the budget, especially after summarization or transformation steps that can change token counts unpredictably.
*   **Over-budget Handling:** Decide how to handle cases where even minimal content exceeds the budget. Return an empty string? A specific warning in the trace?

## Accessing Shared Utilities

Compact Memory provides utilities that can be helpful:

*   **Tokenizers:**
    *   `compact_memory.token_utils.get_tokenizer(tokenizer_name_or_path)`: Helper to load `tiktoken` or `transformers` tokenizers.
    *   `compact_memory.token_utils.token_count(tokenizer, text)`: Counts tokens in a text using the provided tokenizer.
*   **Chunking:**
    *   Strategies now receive an optional ``chunk_fn`` in their `compress` method. This function should conform to the `ChunkFn = Callable[[str], List[str]]` interface.
    *   `examples/chunking.py` contains various example `ChunkFn` implementations that can be used directly or adapted:
        *   `newline_splitter`: Splits text by newline characters.
        *   `tiktoken_fixed_size_splitter`: Splits text into fixed-size chunks based on `tiktoken` token counts.
        *   `langchain_recursive_splitter`: Wraps LangChain's `RecursiveCharacterTextSplitter`.
        *   `agentic_split` and `simple_sentences`: More specialized heuristic-based sentence splitters.
    *   Developers can also provide their own custom chunking functions adhering to the `ChunkFn` type.
*   **LLM Helpers (Optional):**
    *   If your strategy needs to call an LLM, Compact Memory keeps this outside the core package. Check `examples/llm_helpers.py` for lightweight `run_llm()` wrappers that work with small local models or OpenAI.
    *   You can use these helpers directly or swap in your preferred framework (LangChain, AutoGen, etc.). The helpers simply take a prompt and return the generated text.
    *   Remember to manage API keys and errors in your own code when using external providers.

## Structuring Strategy Logic

*   **Modularity:** Keep your compression logic well-organized. Helper methods for distinct steps (e.g., preprocessing, core compression, postprocessing) can improve readability.
*   **Configuration:** If your strategy has tunable parameters (e.g., summarization model, number of sentences to keep), make them arguments to `__init__` with sensible defaults. These parameters should be recorded in the `CompressionTrace`.
*   **State:**
    *   Most strategies should aim to be stateless within the `compress` call for a given input.
    *   If your strategy has *learnable components* (e.g., a fine-tuned model), implement `save_learnable_components` and `load_learnable_components` to manage its state across sessions.

## Testing Your Strategy

Rigorous testing is crucial. Compact Memory's experimentation framework helps with this.

1.  **Unit Tests:**
    *   Write standard Python unit tests for your strategy's core logic. Test edge cases, different input types, and budget handling.
    *   Mock external dependencies like LLM calls if necessary.

2.  **Experimentation Framework:**
    *   Evaluation is now handled externally. Use tools like Promptfoo or UpTrain to benchmark strategies.
    *   See [`EVALUATION.md`](./evaluation.md) for recommended workflows.

3.  **`onboarding_demo.py`:**
    *   The `examples/onboarding_demo.py` script shows a basic example of defining a strategy, registering it, and using it in an experiment. Use it as a reference.

4.  **Packaging for Experiments:**
    *   If you package your strategy (see `docs/SHARING_STRATEGIES.md`), you can include example experiment configurations within your package. The CLI command `compact-memory dev run-package-experiment` can then execute these.

## Registering Your Strategy

For Compact Memory to find and use your strategy, it needs to be registered.

*   **Plugin System:** The preferred way is through the plugin system. If your strategy is part of an installable Python package, you can register it via an entry point in your `pyproject.toml` or `setup.py`. See `docs/SHARING_STRATEGIES.md`.
*   **Direct Registration (for local development/testing):**
    ```python
    from compact_memory.registry import register_compression_strategy
    from .my_strategy_module import MyCustomStrategy

    register_compression_strategy(MyCustomStrategy.id, MyCustomStrategy)
    ```
    This is useful in scripts or during development before packaging.

## Best Practices

*   **Clarity and Simplicity:** Aim for understandable code.
*   **Efficiency:** Be mindful of computational cost, especially if your strategy is complex or calls external services.
*   **Robustness:** Handle potential errors gracefully (e.g., invalid inputs, API failures).
*   **Comprehensive Tracing:** Good traces are invaluable for users and for your own debugging.
*   **Documentation:**
    *   Add detailed docstrings to your strategy class and methods.
    *   If your strategy has unique dependencies or setup requirements, document them in a `README.md` if you package it.

By following this guide, you can effectively contribute new and innovative compression strategies to the Compact Memory ecosystem.
