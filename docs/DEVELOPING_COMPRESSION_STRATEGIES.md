# Developing Compression Strategies

This document is a practical guide for developers on how to implement new `CompressionStrategy` modules within the Compact Memory framework. While `docs/COMPRESSION_STRATEGIES.md` covers the conceptual and theoretical aspects of designing such strategies (the "what" and "why"), this document focuses on the "how-to" – the specific interfaces, methods, and considerations for writing the code.

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

Some compression approaches may incorporate trainable components—for example a small neural summarizer or a policy model that selects which passages to keep. When building such strategies:

1. Encapsulate the learnable model within your `CompressionStrategy` implementation so that the rest of the framework can treat it like any other strategy.
2. Provide `save_learnable_components(path)` and `load_learnable_components(path)` methods to persist and restore model state.
3. Consider how the experimentation framework might drive a simple training loop. For instance, a `ValidationMetric` could supply gradients or rewards that update your summarizer.
4. Document expected resources and dependencies so others can reproduce your results.

### Strategies for Conversational AI and Dynamic Contexts

The `ActiveMemoryManager` shows one pattern for maintaining dialogue context. Learnable strategies might extend this with models that predict relevance scores, or with reinforcement learning to optimize pruning decisions.

## Creating Informative Compression Traces

Every `CompressionStrategy` should return a `CompressionTrace` detailing the
steps performed. Use the standard vocabulary from
`docs/EXPLAINABLE_COMPRESSION.md` for the `type` field and include contextual
information in a `details` dictionary. A minimal example:

```python
trace.steps.append({
    "type": "prune_history_turn",
    "details": {
        "turn_id": "abc-123",
        "text_preview": "User: Yes, that sounds right...",
        "reason_for_action": "lowest_retention_score",
        "retention_score": 0.15,
    },
})
```

These rich traces make strategies easier to debug and analyse. When designing a
new strategy, think about what decisions are being made and record them as steps
in the trace.

## Composing Strategies with Pipelines

`PipelineCompressionStrategy` allows multiple compression steps to be chained
together. Each strategy in the pipeline receives the same token budget and
processes the output of the previous one. Pipelines are configured with
`PipelineStrategyConfig`, which lists the strategies to run in order.

Example:

```yaml
compression:
  strategy_name: pipeline
  strategies:
    - strategy_name: importance
    - strategy_name: learned_summarizer
```
