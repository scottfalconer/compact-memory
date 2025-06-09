# Configuring Compact Memory

Compact Memory offers a flexible configuration system to tailor its behavior to your needs and streamline your command-line usage. You can set default values for commonly used options so you don't have to type them every time.

## Configuration Layers and Precedence

Compact Memory resolves settings from the following sources, in order of highest to lowest precedence:

1.  **Command-Line Arguments:**
    Options provided directly with a command will always take top priority.
    *Example:* `compact-memory query "Hello" --model-id openai/gpt-4` will use `openai/gpt-4` for this query, regardless of other settings.

2.  **Environment Variables:**
    You can set specific environment variables to configure Compact Memory globally for your shell session or system. Any configuration key can be overridden by using the prefix `COMPACT_MEMORY_` followed by the key name in uppercase.
    *   `COMPACT_MEMORY_PATH`: Sets the default path to your memory store.
    *   `COMPACT_MEMORY_DEFAULT_MODEL_ID`: Sets the default model ID for LLM interactions.
    *   `COMPACT_MEMORY_DEFAULT_ENGINE_ID`: Sets the default compression engine ID.
    *Example (bash):* `export COMPACT_MEMORY_PATH="/path/to/my/global_memory"`

3.  **Local Project Configuration (`.gmconfig.yaml`):**
    For project-specific settings, you can create a `.gmconfig.yaml` file in your project's root directory (the directory from which you run `compact-memory` commands).
    This allows different projects to have different default Compact Memory configurations.
    *Example `.gmconfig.yaml` content:*
    ```yaml
    compact_memory_path: ./project_memory_store
    default_model_id: local/codellama-7b
    ```

4.  **User Global Configuration (`~/.config/compact_memory/config.yaml`):**
    This is your personal default configuration file, applied across all your projects unless overridden by a local config, environment variable, or CLI argument. The path might vary slightly by OS (e.g., `~/Library/Application Support/compact_memory/config.yaml` on macOS, `~/.config/compact_memory/config.yaml` on Linux).
    The `compact-memory config set` command modifies this file.
    *Example `~/.config/compact_memory/config.yaml` content:*
    ```yaml
    compact_memory_path: ~/my_default_compact_memory
    default_model_id: openai/gpt-4.1-nano
    default_engine_id: prototype
    # You can also set log_file and verbose here
    # log_file: /path/to/compact_memory.log
    # verbose: false
    ```

5.  **Application Defaults:**
    If a setting is not found in any of the above layers, a hardcoded application default value is used.

## Streamlining Operations with `compact-memory config set`

The most convenient way to manage your user-level global defaults is with the `compact-memory config set` and `compact-memory config show` commands.

### Viewing Current Configuration

To see all your current effective configurations and where they are being loaded from (e.g., default, user config, environment variable), use:

```bash
compact-memory config show
```

This prints a table of all effective configuration values along with the source of each setting.

To view a specific key:

```bash
compact-memory config show --key compact_memory_path
```

### Setting Default Values

You can set default values for key options to avoid typing them repeatedly. These settings are saved to your user global configuration file.

**Known Configuration Keys (manageable via `config set`):**

*   `compact_memory_path`: Your primary memory location.
    ```bash
    compact-memory config set compact_memory_path /path/to/your/main_memory
    ```
    Once set, commands like `compact-memory query ...` will use this path automatically unless you override it with the `--memory-path` option for a specific command.

*   `default_model_id`: Your preferred LLM for queries and other LLM-dependent operations.
    ```bash
    compact-memory config set default_model_id openai/gpt-4-turbo
    ```
    Commands like `compact-memory query` will use this model by default.

*   `default_engine_id`: Your preferred compression engine for summarization or queries (if applicable).
    ```bash
    compact-memory config set default_engine_id prototype
    ```
    Commands like `compact-memory compress` will use this engine by default if you don't specify one with `--engine`.

*(Note: While `log_file` and `verbose` can be set in config files manually, they are primarily controlled via CLI options for runtime flexibility. The `config set` command currently supports `compact_memory_path`, `default_model_id`, and `default_engine_id` as these are the most common global defaults users might want to persist.)*

Plugins may register additional configuration keys at runtime. Any such keys will automatically appear when running `compact-memory config show` and can be set using `config set`.


**Example Workflow:**

1.  **Set your preferred memory path:**
    ```bash
    compact-memory config set compact_memory_path "~/compact_memories"
    ```
    *(The CLI will expand `~` to your home directory).*
2.  **Set your default LLM:**
    ```bash
    compact-memory config set default_model_id "openai/gpt-4.1-nano"
    ```
3.  **Initialize an agent (it will now use the default path if not specified):**
    ```bash
    compact-memory agent init
    # If "~/compact_memories" does not exist, it will be created.
    # If it exists and is a valid agent, this might show an error unless it's empty or a different path is given.
    # Typically, you initialize to a specific new path first:
    # compact-memory agent init ./my_specific_agent_location
    # Then, if desired, set this as the global default:
    # compact-memory config set compact_memory_path ./my_specific_agent_location
    ```
4.  **Now, you can run commands more simply, relying on your global defaults:**
    ```bash
    # Assuming compact_memory_path is set and the agent is initialized at that path
    compact-memory compress --file my_document.txt
    compact-memory query "What was in my document?"
    ```

By effectively using the configuration layers, especially `compact-memory config set` for your global defaults, you can significantly simplify your interactions with the Compact Memory CLI. Remember that command-line options always override these defaults if you need to work with a different agent or setting temporarily.

## Engine Configuration (`EngineConfig`)

While the above sections focus on CLI and environment-level configuration, when working with Compact Memory engines programmatically in Python, you'll interact with the `EngineConfig` class (from `compact_memory.engine_config`). This Pydantic model defines the settings for a compression engine.

Key fields in `EngineConfig` include:

*   `chunker_id`: Identifier for the chunker (e.g., "fixed_size", "sentence_window").
*   `vector_store`: Identifier for the vector store (e.g., "in_memory", "faiss").
*   `embedding_dim`: Dimension of embeddings.
*   `vector_store_path`: Path for persistent vector stores.
*   `embedding_fn_path`: Optional path to a custom embedding function.
*   `preprocess_fn_path`: Optional path to a custom preprocessing function.
*   `enable_trace: bool`: Controls whether compression tracing is active (defaults to `True`).

### Serializing Custom Embedding and Preprocessing Functions

The `embedding_fn_path` and `preprocess_fn_path` fields are crucial for engines that use custom, user-provided Python functions for embedding or text preprocessing. To allow an engine's configuration (and thus its custom functions) to be saved and loaded, you can provide the importable path to these functions.

*   **Purpose**: These fields store the string path (e.g., `my_package.my_module.my_embedding_function`) to your custom functions. When an engine is saved, these paths are persisted. When the engine is loaded, it attempts to re-import these functions using their stored paths.
*   **Importability**: For a custom function to be serializable and reloadable via its path, it **must be defined in a Python module** that can be imported. Lambda functions, inner functions (defined inside another function), or functions in the `__main__` script are generally not importable by path from a different session and thus cannot be serialized this way.
    *   If you provide an importable function programmatically when creating an engine, the engine will attempt to determine its path and store it in the corresponding `*_fn_path` field in its configuration.
    *   If a non-importable function is provided, a warning will be issued, and its path will not be stored, meaning it won't be automatically reloaded if the engine is saved and loaded.
*   **Example**:
    If you have a custom embedding function in `my_custom_utils.py`:
    ```python
    # my_custom_utils.py
    import numpy as np
    def custom_embedder(texts):
        # Your custom embedding logic
        print(f"Custom embedder called with: {texts}")
        return np.random.rand(len(texts) if isinstance(texts, list) else 1, 10) # Example: 10-dim embedding
    ```
    You could configure an engine (programmatically or potentially via a config file if your application supports it) to use this:
    ```python
    from compact_memory.engine_config import EngineConfig
    from compact_memory.engines.base import BaseCompressionEngine # Or a specific engine

    # Programmatic configuration
    config = EngineConfig(
        embedding_fn_path="my_custom_utils.custom_embedder",
        embedding_dim=10 # Important to set if not discoverable by default
        # ... other config settings
    )
    # engine = BaseCompressionEngine(config=config)
    # Now, if 'engine' is saved, 'embedding_fn_path' will be in its manifest.
    # Upon loading, it will try to import 'custom_embedder' from 'my_custom_utils'.
    ```

### Controlling Compression Tracing

The `enable_trace` field in `EngineConfig` allows you to control the generation of `CompressionTrace` objects during an engine's `compress` operation.

*   **Purpose**: `CompressionTrace` objects record detailed information about the steps, decisions, and performance of a compression operation, which is invaluable for debugging, analysis, and explainability. However, generating this trace can have a minor performance overhead.
*   **Behavior**:
    *   If `enable_trace` is set to `True` (which is the default), the `compress` method of an engine will generate a `CompressionTrace` object and include it in the returned `CompressedMemory.trace` field.
    *   If `enable_trace` is set to `False`, the `compress` method will skip the generation of the trace, and `CompressedMemory.trace` will be `None`. The primary compression logic will still be executed.
*   **Use Case**: You might want to set `enable_trace: False` in production or performance-sensitive scenarios where the detailed trace is not needed, to save computation resources. For development, debugging, or when explainability is important, keeping it enabled is recommended.
*   **Example**:
    To disable compression tracing for an engine:
    ```python
    from compact_memory.engine_config import EngineConfig
    from compact_memory.engines.base import BaseCompressionEngine # Or any specific engine

    # Disable tracing
    config_no_trace = EngineConfig(
        enable_trace=False
        # ... other config settings
    )
    engine_no_trace = BaseCompressionEngine(config=config_no_trace)

    # When engine_no_trace.compress(...) is called, the result will have trace=None
    # compressed_memory = engine_no_trace.compress("Some long text...", budget=100)
    # assert compressed_memory.trace is None

    # Enable tracing (default behavior)
    config_with_trace = EngineConfig(
        enable_trace=True # Explicitly True, or omit as it's the default
        # ... other config settings
    )
    engine_with_trace = BaseCompressionEngine(config=config_with_trace)
    # compressed_memory_with_trace = engine_with_trace.compress("Some long text...", budget=100)
    # assert compressed_memory_with_trace.trace is not None
    ```

The `EngineConfig` model uses Pydantic for data validation, so any values you provide (either directly in Python or through a configuration mechanism your application might build on top of it) will be validated against the defined types and constraints.

## Logging

Compact Memory uses Python's standard `logging` module to provide information about its operations. This can be helpful for understanding the internal workings, debugging issues, or monitoring behavior.

### Configuring Logging via CLI

The easiest way to control logging is through global CLI options:

*   **`--verbose` / `-V`**: Enables verbose logging. This sets the logging level to `DEBUG`, providing detailed information about engine operations, including specific steps within compression or data processing.
    ```bash
    compact-memory --verbose engine stats --memory-path ./my_store
    ```

*   **`--log-file PATH`**: Specifies a file path where logs should be written. If this option is not provided, logs are typically sent to standard error (stderr).
    ```bash
    compact-memory --log-file ./cm_debug.log --verbose engine rebuild-index --memory-path ./my_store
    ```
    In this example, detailed DEBUG level logs will be written to `cm_debug.log`.

### Log Levels and Content

*   **INFO Level**: Enabled by default (unless `--verbose` is used). Provides general information about major operations, such as:
    *   Engine initialization.
    *   Start and completion of ingestion, recall, save, load, and index rebuilding operations.
    *   Summary information (e.g., number of items ingested/recalled).
*   **DEBUG Level** (enabled by `--verbose`): Provides much more granular information, including:
    *   Detailed steps within engine `compress` methods (e.g., "Truncating text...", "Applied stopword pruning...").
    *   Specifics of data being processed by vector stores (e.g., "InMemoryVectorStore: Rebuilding FAISS index.").
    *   Configuration details being used by components.
    *   Information helpful for diagnosing unexpected behavior or errors.

### Using Logs

*   **Debugging**: When encountering an error or unexpected behavior, enabling verbose logging to a file (`--verbose --log-file app.log`) is often the first step. The detailed logs can show the sequence of operations and internal states leading up to the issue.
*   **Monitoring**: For long-running processes or automated tasks, logging to a file can provide a record of operations.
*   **Understanding Internals**: DEBUG logs can be a useful way to learn how different engines and components process data.

While `log_file` and `verbose` settings can also be placed in the user's global configuration file (`~/.config/compact_memory/config.yaml`), using the CLI options is generally recommended for more direct control during specific command executions.
```
