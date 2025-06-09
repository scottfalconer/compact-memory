# Developing Compression Engines

The Compact Memory framework allows developers to create and integrate custom compression engines. This guide covers the core concepts and steps involved in building your own engine by subclassing `BaseCompressionEngine`.

## Core Concepts of a `BaseCompressionEngine` Subclass

A compression engine is a Python class that inherits from `compact_memory.engines.BaseCompressionEngine`. It encapsulates the logic for processing, storing (optionally), recalling, and compressing text.

Key aspects to implement or override:

### 1. `id` (Class Attribute)

Each engine must have a unique string `id` as a class attribute. This ID is used for registration and for identifying the engine in configurations and the CLI.

```python
from compact_memory.engines import BaseCompressionEngine

class MyCustomEngine(BaseCompressionEngine):
    id = "my_custom_engine"
    # ... rest of the implementation
```

### 2. `__init__(self, *, chunker=None, embedding_fn=None, preprocess_fn=None, config=None, **kwargs)`

*   **Purpose:** Initializes the engine instance.
*   **Parameters:**
    *   `chunker`: An optional `Chunker` instance (e.g., `SentenceWindowChunker`). If not provided, `BaseCompressionEngine` defaults to `SentenceWindowChunker()`.
    *   `embedding_fn`: An optional callable for generating embeddings. Defaults to `embed_text` from the library.
    *   `preprocess_fn`: An optional callable for text preprocessing.
    *   `config (Optional[Dict[str, Any]])`: A dictionary containing configuration parameters for the engine. This is crucial for making your engine configurable via `load_engine` and the CLI.
    *   `**kwargs`: Can be used to catch additional parameters, often merged with `config`.
*   **Implementation:**
    *   **Call `super().__init__(...)`**: It's important to call the parent class's `__init__` method, passing through relevant parameters like `chunker`, `preprocess_fn`, and crucially, the `config`.
        ```python
        super().__init__(chunker=actual_chunker, preprocess_fn=preprocess_fn, config=config or kwargs)
        ```
    *   **Initialize `self.config`**: `BaseCompressionEngine.__init__` will initialize `self.config` with the `config` dictionary you pass to it. If no `config` is passed to `super().__init__`, `self.config` will be an empty dictionary. If you don't pass a `config` argument to your engine's `__init__` but still want to save parameters, you should populate `self.config` manually in your `__init__`.
    *   **Custom Parameters:** Store any custom parameters (e.g., `similarity_threshold`, `model_name`) as instance attributes. Ideally, these should be derived from the `config` dictionary.
        ```python
        self.my_param = self.config.get('my_param', 'default_value')
        ```
    *   **State Initialization:** Initialize any internal state, data structures, or models your engine needs. For example, if your engine uses a vector store, you might initialize it here, potentially configuring it from `self.config`.

```python
from compact_memory.engines import BaseCompressionEngine
from compact_memory.chunker import SentenceWindowChunker # Example
from typing import Optional, Dict, Any, Callable
import numpy as np

class MyConfigurableEngine(BaseCompressionEngine):
    id = "my_configurable_engine"

    def __init__(
        self,
        *,
        chunker: Optional[Chunker] = None,
        embedding_fn: Optional[Callable[[str | Sequence[str]], np.ndarray]] = None,
        preprocess_fn: Optional[Callable[[str], str]] = None,
        config: Optional[Dict[str, Any]] = None,
        custom_threshold: float = 0.5, # Can be a direct param
        **kwargs
    ):
        # Prepare config for superclass. Merge direct params into config if they should be persisted.
        effective_config = kwargs.copy()
        if config:
            effective_config.update(config)

        # Ensure parameters meant for config are in effective_config
        effective_config.setdefault('custom_threshold', custom_threshold)
        effective_config.setdefault('chunker_id', type(chunker or SentenceWindowChunker()).__name__)

        super().__init__(
            chunker=chunker, # Or SentenceWindowChunker()
            embedding_fn=embedding_fn, # Or specific default
            preprocess_fn=preprocess_fn,
            config=effective_config
        )

        # Initialize attributes from self.config (set by super().__init__)
        self.custom_threshold = float(self.config.get('custom_threshold', 0.5))
        self.internal_data = []

        print(f"MyConfigurableEngine initialized with threshold: {self.custom_threshold}, chunker_id from config: {self.config.get('chunker_id')}")

```

### 3. `self.config` and Persistence

*   The `self.config` dictionary (initialized by `BaseCompressionEngine.__init__`) is automatically saved to `engine_manifest.json` when `super().save(path)` is called.
*   When `load_engine(path)` is used, the `config` from the manifest is passed to your engine's `__init__` method.
*   **What to store in `self.config`:**
    *   Simple, serializable parameters (strings, numbers, booleans, simple lists/dicts).
    *   Identifiers for complex objects (e.g., `chunker_id = type(self.chunker).__name__`). Your engine's `__init__` would then be responsible for reconstructing the chunker from this ID if needed (possibly using a registry or factory pattern).
*   **Updating `self.config` before saving:** If your engine has attributes that should be persisted and might change during its lifecycle, ensure you update `self.config` with their current values before calling `super().save(path)` in your `save` method.

    ```python
    # In your engine's save method, before calling super().save()
    self.config['my_param_to_save'] = self.my_param_to_save
    self.config['another_value'] = self.some_other_attribute
    super().save(path)
    ```

### 4. `ingest(self, text: str) -> List[str]` (Optional Override)

*   **Purpose:** Process and store text in the engine's memory (if it maintains one).
*   **Default Behavior:** The `BaseCompressionEngine.ingest` method chunks the text, computes embeddings for each chunk, and stores the chunks and their embeddings in `self.memories` (a list of dicts) and `self.embeddings` (a NumPy array). It also builds a FAISS index (`self.index`).
*   **When to Override:**
    *   If your engine uses a different storage mechanism (e.g., a custom vector database, graph database).
    *   If you need custom logic for chunking, embedding, or metadata storage during ingestion.
    *   If your engine doesn't maintain a general-purpose memory store in this way (e.g., it's a stateless compressor).
*   **Returns:** A list of IDs for the ingested items/chunks.

### 5. `_compress_chunk(self, chunk_text: str) -> str` (Optional Override)

*   **Purpose:** Compresses a single text chunk. This method is called by `BaseCompressionEngine.ingest` *after* chunking but *before* embedding and storage if you are using the base `ingest` method.
*   **Default Behavior:** Returns the `chunk_text` unmodified (identity compression).
*   **When to Override:** If your engine applies a specific transformation or compression to individual chunks before they are embedded or stored by the base `ingest` logic. For example, an engine might summarize each chunk here.
*   **Returns:** The processed (potentially compressed) text chunk.

### 6. `recall(self, query: str, *, top_k: int = 5) -> List[Dict[str, Any]]` (Optional Override)

*   **Purpose:** Retrieve memories or information relevant to a given query.
*   **Default Behavior:** `BaseCompressionEngine.recall` performs a FAISS similarity search on the `self.embeddings` store built by `ingest`.
*   **When to Override:**
    *   If using a different storage/retrieval mechanism.
    *   To implement custom ranking or filtering logic.
    *   If your engine's recall concept is different (e.g., querying a knowledge graph).
*   **Returns:** A list of dictionaries, where each dictionary represents a retrieved item and should typically include "id", "text", and "score".

### 7. `compress(self, text: str, budget: int, previous_compression_result: Optional[CompressedMemory] = None, **kwargs) -> CompressedMemory` (Typically Required)

*   **Purpose:** This is the primary method for performing one-shot compression of a given text to meet a specified token budget. This method is used by the `compact-memory compress` CLI command and by pipeline engines.
*   **Parameters:**
    *   `text`: The input string to compress. While some engines might internally handle or accept a list of chunks (often named `text_or_chunks`), the standardized signature favors a single string input for simplicity at the interface level.
    *   `budget`: The target token budget for the compressed output (e.g., `llm_token_budget`).
    *   `previous_compression_result: Optional[CompressedMemory] = None`: An optional parameter that provides the output of a preceding compression engine in a chain or pipeline. This allows an engine to base its processing on a previously compressed version of the text. For standalone calls, this is typically `None`.
    *   `**kwargs`: Often includes `tokenizer` (a callable that behaves like a Hugging Face tokenizer, returning a dict with `input_ids`) which can be used to count tokens and truncate accurately. Other engine-specific parameters can also be passed via `kwargs`.
*   **Implementation:** This is where your core compression algorithm resides.
*   **Returns:** A single `CompressedMemory` object. This object now encapsulates all information about the compression result:
    *   `text: str`: The actual compressed text string.
    *   `engine_id: Optional[str]`: The ID of the engine that produced this result (e.g., `self.id`). This should be populated by your engine.
    *   `engine_config: Optional[Dict[str, Any]]`: The configuration of the engine instance that performed the compression (e.g., `self.config` or a relevant subset). This should be populated by your engine.
    *   `trace: Optional[CompressionTrace]`: A `CompressionTrace` object detailing the engine used, parameters, input/output summaries, and steps taken during compression. This object, previously returned as the second element of a tuple, should now be created by your engine and assigned to this field.
    *   `metadata: Optional[Dict[str, Any]]`: A dictionary for any other arbitrary metadata your engine might want to associate with the compressed output.

#### Using `previous_compression_result`

The `previous_compression_result` parameter allows engines to be chained effectively. An engine can inspect this object to:
*   Access the `text` field: `previous_compression_result.text` provides the already compressed text from the prior stage.
*   Examine `metadata`: `previous_compression_result.metadata` might contain useful context.
*   Review the `trace`: `previous_compression_result.trace` can inform decisions.
*   Check `engine_id` or `engine_config`: To understand what kind of compression was previously applied.

For example, an engine could decide to further summarize `previous_compression_result.text`, or it might use some metadata to guide its own parameters. If an engine does not use the previous result, it can simply ignore this parameter.

```python
# Example snippet within a compress method:
if previous_compression_result:
    input_text_for_this_engine = previous_compression_result.text
    # Optionally, adjust behavior based on previous_compression_result.engine_id or metadata
else:
    input_text_for_this_engine = original_text_passed_to_compress
# ... proceed to compress input_text_for_this_engine ...
```

### 8. `save(self, path: str | Path)` (Override for Custom State)

*   **Purpose:** Persist the engine's state to disk.
*   **Implementation:**
    1.  **Update `self.config`:** Ensure any serializable parameters you want to save are in `self.config`.
        ```python
        self.config['my_runtime_param'] = self.my_runtime_param
        ```
    2.  **Call `super().save(path)`:** This saves `engine_manifest.json` (including `self.config`) and, if you used the base `ingest` method, `entries.json` and `embeddings.npy`.
        ```python
        super().save(path) # path is a Path object
        ```
    3.  **Save Custom State:** If your engine has additional state not managed by `BaseCompressionEngine` (e.g., custom model files, its own vector store files), save them to the `path` directory.
        ```python
        custom_data_path = path / "my_custom_data.json"
        # Save your custom data...
        ```
*   **Note:** The `path` provided is a directory where your engine should store all its files.

### 9. `load(self, path: str | Path)` (Override for Custom State)

*   **Purpose:** Load the engine's state from disk.
*   **Implementation:**
    1.  **`self.config` is Pre-populated:** When `load_engine(path)` is used, it first instantiates your engine and passes the `config` from `engine_manifest.json` to your `__init__`. So, `self.config` should already be populated when `load()` is called. Your `__init__` should use this `self.config` to set initial attributes.
    2.  **Call `super().load(path)` (Conditional):**
        *   If your engine relies on `entries.json` and `embeddings.npy` saved by `BaseCompressionEngine.save()`, then call `super().load(path)`.
        *   If your engine manages all its state independently of these files, you might not need to call `super().load(path)`.
    3.  **Load Custom State:** Load any custom files you saved in your `save` method. Use `self.config` if needed to determine how to load/configure custom state.
        ```python
        custom_data_path = path / "my_custom_data.json"
        # Load your custom data and reinitialize parts of your engine...
        ```

## Registration

For your engine to be discoverable by `get_compression_engine(engine_id)` and the CLI, it needs to be registered.

### Manual Registration

You can manually register your engine class:

```python
from compact_memory.engines.registry import register_compression_engine
from .my_engine_module import MyCustomEngine # Assuming your engine is in this module

register_compression_engine(MyCustomEngine.id, MyCustomEngine)
```
This is typically done in your project's `__init__.py` or a dedicated plugin loading module.

### Plugin Entry Points (for Shareable Packages)

If you are developing a separate package for your engine, you can make it discoverable as a plugin using Python's entry points. In your package's `pyproject.toml` (or `setup.py`):

```toml
# pyproject.toml
[project.entry-points."compact_memory.compression_engines"]
my_custom_engine_id = "my_package.my_engine_module:MyCustomEngine"
```

Replace `my_custom_engine_id` with your engine's unique ID, and `my_package.my_engine_module:MyCustomEngine` with the actual import path to your engine class. Compact Memory will automatically discover and register engines defined this way when it loads plugins.

## Example: Simple Truncation Engine

```python
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Sequence, Callable
import numpy as np

from compact_memory.engines import (
    BaseCompressionEngine,
    CompressedMemory,
    CompressionTrace,
)
from compact_memory.engines.registry import register_compression_engine

class SimpleTruncEngine(BaseCompressionEngine):
    id = "simple_trunc"

    def __init__(self, *, config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(config=config or kwargs)
        # No custom state beyond what BaseCompressionEngine handles by default
        # self.config might contain 'chunker_id' if passed.

    # No need to override ingest or recall if we don't store memory
    # No need to override _compress_chunk if not using base ingest's pre-embedding step

    def compress(
        self,
        text: str, # Changed from text_or_chunks for typical signature
        budget: int, # Changed from llm_token_budget
        previous_compression_result: Optional[CompressedMemory] = None, # Added
        **kwargs
    ) -> CompressedMemory: # Changed return type

        text_to_compress = text # Assuming text is already a string

        # Example of how previous_compression_result might be used (optional)
        # if previous_compression_result:
        #     text_to_compress = previous_compression_result.text
            # Or, combine/modify based on previous_compression_result.metadata, etc.

        tokenizer = kwargs.get("tokenizer")
        original_length_chars = len(text_to_compress)
        original_length_tokens = None
        if tokenizer:
            try:
                original_length_tokens = len(tokenizer(text_to_compress)['input_ids'])
            except Exception: # Fallback if tokenizer fails
                pass

        # Simple character-based truncation for this example
        # A real engine would use the tokenizer for token-based truncation
        estimated_chars_per_token = 4 # Example value
        char_budget = budget * estimated_chars_per_token # Use 'budget'
        compressed_text = text_to_compress[:char_budget]

        compressed_length_chars = len(compressed_text)
        compressed_length_tokens = None
        if tokenizer:
            try:
                compressed_length_tokens = len(tokenizer(compressed_text)['input_ids'])
            except Exception:
                pass

        current_trace = CompressionTrace( # Renamed 'trace' to 'current_trace' to avoid conflict if 'trace' is a field
            engine_name=self.id,
            strategy_params={"budget": budget, "method": "character_truncation"}, # Use 'budget'
            input_summary={
                "original_length_chars": original_length_chars,
                "original_length_tokens": original_length_tokens,
            },
            steps=[
                {
                    "type": "truncation",
                    "details": {
                        "original_length_chars": original_length_chars,
                        "char_budget": char_budget,
                        "final_length_chars": compressed_length_chars,
                    },
                }
            ],
            output_summary={
                "compressed_length_chars": compressed_length_chars,
                "compressed_length_tokens": compressed_length_tokens,
            },
            final_compressed_object_preview=compressed_text[:100], # Preview of the result
        )

        return CompressedMemory(
            text=compressed_text,
            engine_id=self.id,
            engine_config=self.config, # Populate with engine's config
            trace=current_trace,      # Embed the trace object
            metadata={"notes": "Simple truncation example"} # Optional metadata
        )

    # No custom save/load needed if only relying on config persistence via superclass
    # and not storing additional state.

# To make it discoverable if run in a context where this file is imported:
# register_compression_engine(SimpleTruncEngine.id, SimpleTruncEngine)
```

This example demonstrates a basic engine that performs simple truncation. More sophisticated engines would implement more complex logic in `compress`, and potentially override other methods like `ingest`, `recall`, `save`, and `load` to manage specialized state or behavior. Remember to update `self.config` appropriately in `__init__` and `save` if you want parameters to be persisted and reloaded.
