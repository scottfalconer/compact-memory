# Developing Compression Strategies

This guide provides a comprehensive walkthrough for researchers and developers looking to create new `CompressionStrategy` implementations within the Compact Memory framework. It covers the core concepts, practical steps, and best practices for building, testing, and integrating your custom strategies.

## Implementing a `CompressionStrategy` for the Python API

When developing a new compression strategy to be used with the `CompactMemoryAgent` or the stateless `compress_text` function, there are key integration points to consider.

### 1. Strategy Class Definition and Constructor

Your custom strategy should inherit from `compact_memory.compression.strategies_abc.CompressionStrategy`.

The constructor of your strategy class will typically receive its specific parameters, and optionally, pre-initialized LLM provider and tokenizer instances if your strategy requires them. A common constructor pattern is:

```python
from compact_memory.compression.strategies_abc import CompressionStrategy
from compact_memory.llm_providers_abc import LLMProvider # Abstract Base Class
from compact_memory.token_utils import Tokenizer # Type for tokenizer instance
from typing import Dict, Optional, Any, Tuple, List, Union # Added List, Union
from compact_memory.models import CompressedMemory as InternalCompressedMemory # Internal model
from compact_memory.compression.trace import CompressionTrace # Internal model


class MyCustomStrategy(CompressionStrategy):
    id = "MyCustomStrategy" # Unique class identifier for registration

    # Optional: Declare if your strategy typically needs an LLM or Tokenizer
    requires_llm = False
    requires_tokenizer = False

    def __init__(self,
                 params: Optional[Dict[str, Any]] = None,
                 llm_provider: Optional[LLMProvider] = None,
                 tokenizer: Optional[Tokenizer] = None):
        super().__init__(params=params, llm_provider=llm_provider, tokenizer=tokenizer) # Call super's __init__

        # Store parameters and dependencies
        self.my_param = self.params.get("my_specific_parameter", "default_value")
        # self.llm_provider and self.tokenizer are already set by super().__init__

        # Initialize any other state your strategy needs
        # For example, load a small model or set up internal variables
        # if self.llm_provider:
        #     print(f"Strategy {self.id} initialized with LLM: {self.llm_provider.config.model_name}")
        # if self.tokenizer:
        #     print(f"Strategy {self.id} initialized with Tokenizer: {self.tokenizer.name}")
        print(f"Strategy {self.id} initialized with param: {self.my_param}")


    def compress(self,
                 text_or_chunks: Union[str, List[str]],
                 budget: int,
                 **kwargs) -> Tuple[InternalCompressedMemory, CompressionTrace]:
        # `kwargs` may also contain 'tokenizer' and 'llm_provider' if passed by the caller
        # (e.g., by `compress_text` function).
        # Your strategy can decide to use these per-call instances or the ones from __init__.
        tokenizer_to_use = kwargs.get('tokenizer', self.tokenizer)
        llm_to_use = kwargs.get('llm_provider', self.llm_provider)

        # --- Your compression logic here ---
        original_text_for_trace = text_or_chunks if isinstance(text_or_chunks, str) else " ".join(text_or_chunks)
        original_tokens = 0
        if tokenizer_to_use:
            original_tokens = tokenizer_to_use.count_tokens(original_text_for_trace)

        # Example: Simple truncation or selection based on `my_param`
        compressed_text_output = f"Compressed based on '{self.my_param}': {original_text_for_trace[:budget*5]}" # Crude budget use

        if llm_to_use and self.my_param == "use_llm_for_summary":
            # compressed_text_output = llm_to_use.generate_response(f"Summarize: {original_text_for_trace}", max_tokens=budget)
            pass # Placeholder for actual LLM call

        # --- End of compression logic ---

        # Create internal models for return
        internal_cm = InternalCompressedMemory(text=compressed_text_output)
        # Populate internal_cm.source_references if your strategy creates meaningful links
        # For example, if compressing multiple chunks, reference which ones were used.
        # internal_cm.source_references = [InternalSourceReference(...)]

        trace = CompressionTrace(
            strategy_name=self.id,
            original_tokens=original_tokens,
            # Calculate compressed_tokens accurately if tokenizer_to_use is available
            compressed_tokens=tokenizer_to_use.count_tokens(compressed_text_output) if tokenizer_to_use else 0
        )
        # Populate trace.steps, trace.llm_input, trace.llm_output etc.
        trace.add_step("parameter_check", {"my_param_value": self.my_param, "budget_type": "tokens", "budget_value": budget})
        if llm_to_use and self.my_param == "use_llm_for_summary":
            trace.set_llm_io("Example LLM Input", "Example LLM Output")

        return internal_cm, trace
```

*   **`id` (Class Attribute)**: Your strategy class **must** have a unique `id` class attribute (e.g., `id = "MyCoolStrategy"`). This ID is used by the registry system to look up and instantiate your strategy class based on `strategy_class_id` from a `StrategyConfig`.
*   **`__init__(self, params: Optional[Dict[str, Any]] = None, llm_provider: Optional[LLMProvider] = None, tokenizer: Optional[Tokenizer] = None)`**:
    *   The constructor receives `params` (from `StrategyConfig.params`), an optional `llm_provider` instance, and an optional `tokenizer` instance.
    *   It's **highly recommended** to call `super().__init__(params=params, llm_provider=llm_provider, tokenizer=tokenizer)`. The base `CompressionStrategy` constructor stores these as `self.params`, `self.llm_provider`, and `self.tokenizer`, making them readily available.
    *   You can then access specific parameters using `self.params.get("my_key", default_value)`.
*   **Dependency Injection**: The `CompactMemoryAgent` or `compress_text` function handles the instantiation of LLM providers and tokenizers based on the `StrategyConfig` (or agent defaults) and passes these initialized instances to your strategy's constructor. This means your strategy doesn't need to handle API keys or model loading for these shared components directly if they are injected.

### 2. Implementing the `compress` Method

The core logic of your strategy resides in the `compress` method:

```python
def compress(self,
             text_or_chunks: Union[str, List[str]],
             budget: int,
             **kwargs) -> Tuple[InternalCompressedMemory, CompressionTrace]:
```
*   **`text_or_chunks: Union[str, List[str]]`**: The input to be compressed. This can be a single string of text or a list of pre-chunked strings. Your strategy should be designed to handle the expected input type.
*   **`budget: int`**: The target budget for the compressed output. The interpretation of this budget (e.g., max tokens, number of chunks, character count) is defined by your strategy, but it typically refers to a token limit. Your strategy should try to adhere to this budget.
*   **`**kwargs`**: A dictionary for additional keyword arguments. The `CompactMemoryAgent` and `compress_text` function may pass `tokenizer` and `llm_provider` instances here as well (e.g., `kwargs.get('tokenizer')`). This offers flexibility: your strategy can use the instances stored during `__init__` (`self.tokenizer`, `self.llm_provider`) or prioritize those passed directly to `compress` if a specific per-call instance is needed.
*   **Return Value**: The method **must** return a tuple containing two objects:
    1.  An instance of `compact_memory.models.InternalCompressedMemory`. This internal model object should be populated with:
        *   `text`: The resulting compressed string.
        *   `source_references`: (Optional) A list of `InternalSourceReference` objects if your strategy can link parts of the compressed text back to original chunks or documents.
    2.  An instance of `compact_memory.compression.trace.CompressionTrace`. This object should be populated with details about the compression process, including the strategy name, original and compressed token counts (if applicable), any LLM interactions, and a list of steps taken. (See "Creating Informative Compression Traces" section below).

### 3. Declaring LLM/Tokenizer Requirements (Optional but Recommended)

To help users (and potentially automated configuration tools) understand your strategy's dependencies, you can declare whether it typically requires an LLM or a tokenizer by adding boolean class attributes:

```python
class MyCustomStrategy(CompressionStrategy):
    id = "MyCustomStrategy"
    requires_llm = True      # Indicates this strategy generally needs an LLM.
    requires_tokenizer = True # Indicates this strategy generally needs a tokenizer.

    # ... __init__ and compress methods ...
```
The `compress_text` stateless function uses these flags to raise a `ConfigurationError` early if a required component is not provided. The `CompactMemoryAgent` might also use these for validation or to provide more helpful error messages if a strategy fails to initialize due to missing dependencies.

### 4. Registering Your Strategy

For your strategy to be discoverable by its `strategy_class_id` (as specified in `StrategyConfig`), it needs to be registered with the system. The primary mechanism for this is through Python's import system:

*   **Import-Based Registration**: The simplest way is to ensure that the Python module containing your custom strategy class is imported at some point before the `CompactMemoryAgent` is initialized or `compress_text` is called with your strategy's ID. When the module containing your strategy class (which should have the `id` class attribute) is imported, it typically registers itself with the central strategy registry (managed by `compact_memory.registry`).

*   **Plugin Systems (Advanced)**: For larger extensions or community contributions, Compact Memory might support a plugin system using entry points defined in `pyproject.toml`. If this is the case, refer to the specific packaging and plugin documentation for `compact-memory`.

For most custom development integrated directly into a project using `compact-memory`, ensuring your strategy module is imported (e.g., `from .my_strategies import MyCustomStrategy`) is usually sufficient.

## Designing Learnable Compression Strategies

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
        text_or_chunks: Union[str, List[str]],
        llm_token_budget: int,
        **kwargs: Any,
    ) -> Tuple[CompressedMemory, CompressionTrace]:
        """
        Compresses the input text or chunks to meet the token budget.

        Args:
            text_or_chunks: Either a single string of text or a list of pre-chunked strings.
                            Your strategy needs to handle both cases or define its expected input.
            llm_token_budget: The target maximum number of tokens (or a proxy like characters,
                            depending on your strategy's design) that the compressed output
                            should ideally have.
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

1.  **Input (`text_or_chunks`):**
    *   Decide if your strategy works best with a single block of text or pre-chunked text.
    *   If you expect chunks, you might need to join them or process them individually.
    *   If you receive a single string, you might need to implement chunking logic within your strategy or use a provided chunker.

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
    *   While strategies can implement their own chunking, Compact Memory also has chunking utilities (e.g., `SentenceWindowChunker`) that can be used externally to prepare input for your strategy or internally if your strategy requires chunk-based processing. See `compact_memory.chunker`.
*   **LLM Providers (Advanced):**
    *   If your strategy involves calling an LLM (e.g., for abstractive summarization), you can leverage Compact Memory's LLM provider abstractions (`compact_memory.llm_providers_abc.LLMProvider`, with implementations like `OpenAIProvider`, `GeminiProvider`, `LocalTransformersProvider`).
    *   This typically involves:
        1.  Accepting LLM configuration (model name, API keys if needed) in your strategy's `__init__`.
        2.  Instantiating the chosen provider.
        3.  Using its `generate_response()` method.
    *   Ensure your strategy handles API errors gracefully and documents LLM dependencies.

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
    *   Compact Memory provides tools to run experiments comparing strategies. This typically involves:
        *   A dataset (e.g., a collection of text files).
        *   One or more compression strategies to test.
        *   Configuration for each strategy (parameters, token budgets).
        *   Validation metrics to evaluate the output.
    *   **Key components:**
        *   `ExperimentConfig`, `ResponseExperimentConfig`, `HistoryExperimentConfig`: Dataclasses for defining experiment parameters.
        *   `run_experiment`, `run_response_experiment`, `run_history_experiment`: Functions to execute these experiments.
        *   `ValidationMetric`: Base class for metrics that evaluate compression quality or task performance (e.g., ROUGE scores, LLM-based evaluation). See `docs/DEVELOPING_VALIDATION_METRICS.md`.
    *   **Workflow:**
        1.  Define an experiment configuration file (often YAML) or create config objects programmatically.
        2.  Specify your strategy's ID and any parameters in the config.
        3.  Run the experiment using the CLI (`compact-memory experiment run ...`) or Python API.
        4.  Analyze the output metrics to see how your strategy performs.

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
