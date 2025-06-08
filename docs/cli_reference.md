# CLI Reference

The `compact-memory` command-line interface (CLI) provides tools for managing engine stores, compressing text, performing queries, and accessing development utilities.

## Global Options

These options can be used with most `compact-memory` commands:

*   `--log-file PATH`: Path to write debug logs.
*   `--verbose, -V`: Enable verbose (DEBUG level) logging.
*   `--memory-path TEXT, -m TEXT`: Path to the Compact Memory engine store directory. Overrides environment variables and configuration files.
*   `--model-id TEXT`: Default model ID for LLM interactions (e.g., for the `query` command).
*   `--engine TEXT`: Default compression engine ID to be used (e.g., for history compression in `query`, or as the default one-shot compressor in `compress`).
*   `--version, -v`: Show application version and exit.
*   `--install-completion`: Install shell completion.
*   `--show-completion`: Show shell completion script.

## Main Commands

### `compact-memory engine`

Group of commands for managing engine stores. Replaces the old `compact-memory memory` group.

#### `compact-memory engine init TARGET_DIRECTORY`

Initializes a new Compact Memory engine store in the specified directory.

*   **`TARGET_DIRECTORY`**: (Required) Directory to initialize the new engine store. Will be created if it doesn't exist.
*   **Options:**
    *   `--engine, -e TEXT`: The ID of the compression engine to initialize (e.g., `none`, `base`). If not provided, uses the global default engine ID, or falls back to "none".
    *   `--name TEXT`: A descriptive name for the engine store or its configuration (default: "default\_store"). This is stored in the engine's configuration.
    *   `--chunker TEXT`: Identifier for the chunker to be used by the engine (default: "SentenceWindowChunker"). This is stored as `chunker_id` in the engine's configuration.
    *   `--help`: Show help message.

*   **Usage Examples:**
    ```bash
    compact-memory engine init ./my_store --engine none --name "My Research Store"
    compact-memory engine init ./my_base_store --engine base --chunker "MyCustomChunkerID"
    ```

#### `compact-memory engine list`

Lists all available (registered) compression engine IDs.

#### `compact-memory engine info ENGINE_ID`

Shows detailed metadata for a specific compression engine ID.

*   **`ENGINE_ID`**: (Required) The ID of the engine to inspect.

#### `compact-memory engine stats`

Displays statistics about an engine store.

*   **Options:**
    *   `--memory-path TEXT, -m TEXT`: Path to the engine store directory. Overrides global setting.
    *   `--json`: Output statistics in JSON format.

#### `compact-memory engine validate`

Validates the integrity of an engine store's storage. (Currently provides a basic check).

*   **Options:**
    *   `--memory-path TEXT, -m TEXT`: Path to the engine store directory. Overrides global setting.

#### `compact-memory engine clear`

Deletes all data from an engine store. This action is irreversible.

*   **Options:**
    *   `--memory-path TEXT, -m TEXT`: Path to the engine store directory. Overrides global setting.
    *   `--force, -f`: Force deletion without prompting for confirmation.
    *   `--dry-run`: Simulate deletion and show what would be deleted (currently indicates no action for non-persistent parts).

### `compact-memory query QUERY_TEXT`

Queries a Compact Memory engine store and returns an AI-generated response. This typically requires an engine that supports querying (like `PrototypeEngine`) and an LLM.

*   **`QUERY_TEXT`**: (Required) The query text to send to the engine.
*   **Options:**
    *   `--memory-path TEXT, -m TEXT`: (Required if not set globally) Path to the engine store to query.
    *   `--show-prompt-tokens`: Display the token count of the final prompt sent to the LLM.
    *   Global options like `--model-id` (for the LLM) and `--engine` (for history compression, if applicable) are relevant here.

*   **Usage Example:**
    ```bash
    compact-memory query --memory-path ./my_store "What is the capital of France?"
    ```

### `compact-memory compress`

Compresses text using a specified compression engine. Can read from a string, file, or directory. The output can be sent to stdout, a file, or ingested into an existing engine store.

*   **Input Options (choose one):**
    *   `--text TEXT`: Raw text to compress, or '-' to read from stdin.
    *   `--file FILE_PATH`: Path to a single text file.
    *   `--dir DIR_PATH`: Path to a directory of input files.
*   **Core Options:**
    *   `--engine, -e TEXT`: (Required if no global default) The ID of the **one-shot compression engine** to use for compressing the input text (e.g., `none`, `dummy_trunc`, or a custom summarization engine).
    *   `--budget INTEGER`: (Required) Token budget for the compressed output. The one-shot engine will aim to keep its output within this limit.
*   **Output Options (choose one or none for stdout):**
    *   `--output, -o FILE_PATH`: File path to write compressed output. Not valid with `--dir`.
    *   `--output-dir DIR_PATH`: Directory to write compressed files when `--dir` is used.
    *   `--json`: Output compressed result (and basic stats) in JSON format to stdout. Not valid with `--memory-path`.
    *   `--memory-path TEXT, -m TEXT`: Path to an **existing engine store**. If specified, the text compressed by the one-shot `--engine` will be **ingested** into this target engine store using its `ingest()` method. No other output options (`--output`, `--output-dir`, `--json`) can be used with `--memory-path`.
*   **Other Options:**
    *   `--output-trace FILE_PATH`: File path to write the `CompressionTrace` JSON object. Not applicable for directory input or when `--memory-path` is used.
    *   `--recursive, -r`: Process text files in subdirectories recursively when `--dir` is used.
    *   `--pattern, -p TEXT`: File glob pattern to match files when `--dir` is used (default: `*.txt`).
    *   `--verbose-stats`: Show detailed token counts and processing time per item.

*   **Usage Examples:**
    *   **Compress text to stdout:**
        ```bash
        compact-memory compress --text "Some very long text..." --engine dummy_trunc --budget 100
        ```
    *   **Compress a file and save to another file:**
        ```bash
        compact-memory compress --file path/to/document.txt -e some_summarizer --budget 200 -o summary.txt
        ```
    *   **Compress all `.md` files in a directory and save to an output directory:**
        ```bash
        compact-memory compress --dir input_dir/ -e none --budget 500 --output-dir output_dir/ --recursive -p "*.md"
        ```
    *   **Compress text and ingest into an existing engine store:**
        ```bash
        # First, ensure the store exists (e.g., by running 'engine init')
        compact-memory engine init ./my_main_store --engine none
        # Now, compress some text using 'dummy_trunc' and ingest the result into 'my_main_store'
        compact-memory compress --memory-path ./my_main_store --text "This is a long text to be truncated and then ingested." --engine dummy_trunc --budget 50
        ```

### `compact-memory config`

Group of commands for managing Compact Memory application configuration settings.

#### `compact-memory config set KEY VALUE`

Sets a configuration key to a new value in the user's global config file (`~/.config/compact_memory/config.yaml`).

*   **`KEY`**: (Required) The configuration key to set (e.g., `default_engine_id`, `compact_memory_path`).
*   **`VALUE`**: (Required) The new value for the key.

#### `compact-memory config show`

Displays current configuration values, their effective settings, and their sources (e.g., environment variable, config file, default).

*   **Options:**
    *   `--key, -k TEXT`: Specific configuration key to display.

### `compact-memory dev`

Group of commands for compression engine developers and researchers.

#### `compact-memory dev list-metrics`

Lists all available validation metric IDs that can be used in evaluations.

Example output:

```
Available validation metric IDs:
- rouge_hf
- bleu_hf
- meteor_hf
- bertscore_hf
- exact_match
- compression_ratio
- embedding_similarity
- llm_judge
```

#### `compact-memory dev list-engines` (also `list-strategies`)

Lists all available compression engine IDs, their versions, and sources (built-in or plugin).

*   **Options:**

#### `compact-memory dev inspect-engine ENGINE_NAME`

Placeholder command for inspecting engine data. Not currently implemented.

*   **`ENGINE_NAME`**: (Required) Name of the engine.
*   **Options:**
    *   `--list-prototypes`: Ignored placeholder option.

#### `compact-memory dev evaluate-compression ORIGINAL_INPUT COMPRESSED_INPUT`

Evaluates compressed text against original text using a specified metric. Input can be direct text, file paths, or '-' for stdin.

*   **`ORIGINAL_INPUT`**: (Required) Original text/file/-.
*   **`COMPRESSED_INPUT`**: (Required) Compressed text/file/-.
*   **Options:**
    *   `--metric, -m TEXT`: (Required) ID of the validation metric.
    *   `--metric-params JSON_STRING`: Metric parameters as a JSON string.
    *   `--json`: Output scores in JSON format.

#### `compact-memory dev test-llm-prompt`

Tests a Language Model (LLM) prompt with specified context and query.

*   **Options:**
    *   `--context, -c TEXT`: Context string, file path, or '-'. (Required)
    *   `--query, -q TEXT`: User query. (Required)
    *   `--model TEXT`: Model ID for the test (default: "tiny-gpt2").
    *   `--system-prompt, -s TEXT`: Optional system prompt.
    *   `--max-new-tokens INTEGER`: Max new tokens for LLM (default: 150).
    *   `--output-response FILE_PATH`: File to save LLM's raw response.
    *   `--llm-config FILE_PATH`: Path to LLM configuration YAML (default: `llm_models_config.yaml`).
    *   `--api-key-env-var TEXT`: Environment variable name for API key.

#### `compact-memory dev evaluate-llm-response RESPONSE_INPUT REFERENCE_INPUT`

Evaluates an LLM's response against a reference answer using a specified metric.

The `llm_judge` metric requires an OpenAI API key available in the
`OPENAI_API_KEY` environment variable. If the key is missing, the metric
raises a clear `RuntimeError`.

*   **`RESPONSE_INPUT`**: (Required) LLM's response text/file/-.
*   **`REFERENCE_INPUT`**: (Required) Reference answer text/file/-.
*   **Options:** Similar to `evaluate-compression` (`--metric`, `--metric-params`, `--json`).

#### `compact-memory dev evaluate-engines`

Runs multiple compression engines on the same input text and reports the
`compression_ratio` and `embedding_similarity` metrics for each one.

*   **Options:**
    *   `--text TEXT`: Raw text to compress, or `-` to read from stdin.
    *   `--file FILE_PATH`: Path to a text file to compress.
    *   `--engine, -e TEXT`: Engine ID to evaluate. Can be repeated; defaults to all engines.
    *   `--budget INTEGER`: Token budget for compression (default: 100).
    *   `--output, -o FILE_PATH`: Optional path to write the JSON results.

#### `compact-memory dev download-embedding-model`

Downloads a specified SentenceTransformer embedding model from Hugging Face.

*   **Options:**
    *   `--model-name TEXT`: Name of the model (default: "all-MiniLM-L6-v2").

#### `compact-memory dev download-chat-model`

Downloads a specified causal Language Model (e.g., for chat) from Hugging Face.

*   **Options:**
    *   `--model-name TEXT`: Name of the Hugging Face model (default: "tiny-gpt2").

#### `compact-memory dev create-engine-package`

Creates a new compression engine extension package from a template.

*   **Options:**
    *   `--name TEXT`: Name for the new engine package (default: "compact\_memory\_example\_engine").
    *   `--path PATH`: Directory where the package will be created.

#### `compact-memory dev validate-engine-package PACKAGE_PATH`

Validates the structure and manifest of a compression engine extension package.

*   **`PACKAGE_PATH`**: (Required) Path to the root directory of the engine package.

#### `compact-memory dev inspect-trace TRACE_FILE`

Inspects a `CompressionTrace` JSON file.

*   **`TRACE_FILE`**: (Required) Path to the trace JSON file.
*   **Options:**
    *   `--type TEXT`: Filter trace steps by this type string.
