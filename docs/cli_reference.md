# Compact Memory CLI Reference

This document provides a comprehensive reference for the Compact Memory Command Line Interface (CLI).

## Global Options

These options can be used with any command:

*   `--memory-path TEXT`: Path to the Compact Memory memory directory. Overrides environment variables and configuration files. (e.g., `-m ./my_memory_data`)
*   `--model-id TEXT`: Default model ID for LLM interactions (e.g., `openai/gpt-4-turbo`). Overrides environment variables and configuration files.
*   `--engine TEXT`: Default compression engine ID (e.g., `prototype`). Overrides environment variables and configuration files.
*   `--log-file PATH`: Path to write debug logs. If not set, logs are not written to file.
*   `--verbose / -V`: Enable verbose (DEBUG level) logging to console and log file (if specified).
*   `--version / -v`: Show the application version and exit.
*   `--help`: Show help messages for any command or subcommand.

## Configuration System

Compact Memory uses a hierarchical configuration system. Settings are resolved in the following order of precedence (highest first):

1.  **Command-Line Arguments:** Options provided directly with a command (e.g., `--memory-path ./my_memory`).
2.  **Environment Variables:**
    *   `COMPACT_MEMORY_PATH`
    *   `COMPACT_MEMORY_DEFAULT_MODEL_ID`
    *   `COMPACT_MEMORY_DEFAULT_ENGINE_ID`
3.  **Local Project Configuration:** A `.gmconfig.yaml` file in the current working directory.
4.  **User Global Configuration:** A `config.yaml` file located in `~/.config/compact_memory/config.yaml` (path may vary slightly by OS).
5.  **Application Defaults:** Hardcoded default values within the application.

You can manage your global configuration using the `compact-memory config` commands. See the [Configuration Guide](configuration.md) for more details.

## Top-Level Commands

These are the primary commands for interacting with Compact Memory.

### `compact-memory ingest`

Ingests a text file or files in a directory into an engine store. The store is determined by the active `--memory-path` or its configured default.

**Usage:** `compact-memory ingest [OPTIONS] SOURCE`

**Arguments:**
*   `SOURCE`: Path to the text file or directory containing text files to ingest. (Required)

**Options:**
*   `--tau FLOAT`: Similarity threshold (0.5-0.95) for memory consolidation. Overrides the store's existing tau if set. If the store is new, this tau is used for initialization.
*   `--json`: Output ingestion summary statistics in JSON format.

### `compact-memory query`

Queries an engine store (specified by `--memory-path` or configuration) with the provided text and returns an AI-generated response.

**Usage:** `compact-memory query [OPTIONS] QUERY_TEXT`

**Arguments:**
*   `QUERY_TEXT`: The query text to send to the engine store. (Required)

**Options:**
*   `--show-prompt-tokens`: Display the token count of the final prompt sent to the LLM.

### `compact-memory compress`

Compresses text content from a string, file, or directory using a specified engine and token budget. This is a standalone utility.

**Usage:** `compact-memory compress [OPTIONS] (--text TEXT | --file PATH | --dir PATH)`

**Input Options (choose exactly one):**
*   `--text TEXT`: Raw text to compress. Use `-` to read from stdin.
*   `--file PATH`: Path to a single text file.
*   `--dir PATH`: Path to a directory of files.

**Options:**
*   `--engine / -s TEXT`: compression engine ID to use. Overrides the global default engine. (Required if no global default is set)
*   `--budget INTEGER`: Token budget for the compressed output. The engine will aim to keep the output within this limit. (Required)
*   `--output / -o PATH`: File path to write compressed output when using `--text` or `--file`. Prints to console if unspecified.
*   `--output-dir PATH`: Directory to write compressed files when using `--dir`.
*   `--output-trace PATH`: File path to write the `CompressionTrace` JSON object. (Not valid with `--dir`).
*   `--recursive / -r`: Process text files in subdirectories recursively (valid with `--dir` only).
*   `--pattern / -p TEXT`: File glob pattern for directory input (valid with `--dir` only; default: "*.txt").
*   `--verbose-stats`: Show detailed token counts and processing time per item.


## Command Groups

### `compact-memory engine`

Manage engine stores: initialize, inspect statistics, validate, and clear.

The engine namespace is used to manage a named engine store for stored, compressed memories. It is not an AI agent.

**Usage:** `compact-memory engine [OPTIONS] COMMAND [ARGS]...`

#### `compact-memory engine init`

Creates and initializes a new engine store in a specified directory.

**Usage:** `compact-memory engine init [OPTIONS] TARGET_DIRECTORY`

**Arguments:**
*   `TARGET_DIRECTORY`: Directory to initialize the new engine store in. Will be created if it doesn't exist. (Required)

**Options:**
*   `--name TEXT`: A descriptive name for the engine store (default: "default").
*   `--model-name TEXT`: Name of the sentence-transformer model for embeddings (default: "all-MiniLM-L6-v2").
*   `--tau FLOAT`: Similarity threshold (tau) for memory consolidation, between 0.5 and 0.95 (default: 0.8).
*   `--alpha FLOAT`: Alpha parameter, controlling the decay rate for memory importance (default: 0.1).
*   `--chunker TEXT`: Chunking engine to use for processing text during ingestion (default: "sentence_window").

#### `compact-memory engine stats`

Displays statistics about the Compact Memory engine store.

**Usage:** `compact-memory engine stats [OPTIONS]`

**Options:**
*   `--memory-path TEXT`: Path to the engine store directory. Overrides global setting if provided.
*   `--json`: Output statistics in JSON format.

#### `compact-memory engine validate`

Validates the integrity of the engine store's storage.

**Usage:** `compact-memory engine validate [OPTIONS]`

**Options:**
*   `--memory-path TEXT`: Path to the engine store directory. Overrides global setting if provided.

#### `compact-memory engine clear`

Deletes all data from an engine store. This action is irreversible.

**Usage:** `compact-memory engine clear [OPTIONS]`

**Options:**
*   `--memory-path TEXT`: Path to the engine store directory. Overrides global setting if provided.
*   `--force / -f`: Force deletion without prompting for confirmation.
*   `--dry-run`: Simulate deletion and show what would be deleted without actually removing files.

### `compact-memory config`

Manage Compact Memory application configuration settings.

**Usage:** `compact-memory config [OPTIONS] COMMAND [ARGS]...`

#### `compact-memory config set`

Sets a Compact Memory configuration key to a new value in the user's global config file (`~/.config/compact_memory/config.yaml`).

**Usage:** `compact-memory config set [OPTIONS] KEY VALUE`

**Arguments:**
*   `KEY`: The configuration key to set (e.g., 'compact_memory_path', 'default_model_id'). (Required)
*   `VALUE`: The new value for the configuration key. (Required)

#### `compact-memory config show`

Displays current Compact Memory configuration values, their effective settings, and their sources.

**Usage:** `compact-memory config show [OPTIONS]`

**Options:**
*   `--key / -k TEXT`: Specific configuration key to display.

### `compact-memory dev`

Developer tools for testing, evaluation, engine/package management, and model downloads.

**Usage:** `compact-memory dev [OPTIONS] COMMAND [ARGS]...`

#### `compact-memory dev list-metrics`
Lists all available validation metric IDs that can be used in evaluations.
**Usage:** `compact-memory dev list-metrics`

#### `compact-memory dev list-engines`
Lists all available compression engine IDs, their versions, and sources (built-in or plugin).
**Usage:** `compact-memory dev list-engines`

#### `compact-memory dev inspect-engine`
Inspects aspects of a compression engine, currently focused on 'prototype' engine's beliefs.
**Usage:** `compact-memory dev inspect-engine [OPTIONS] ENGINE_ID`
**Arguments:**
*   `ENGINE_ID`: The ID of the engine to inspect. Currently, only 'prototype' is supported. (Required)
**Options:**
*   `--memory-path TEXT`: Path to the engine store directory. Overrides global setting if provided. Required if '--list-prototypes' is used.
*   `--list-prototypes`: List consolidated prototypes (beliefs) if the engine is 'prototype' and a memory path is provided.

#### `compact-memory dev evaluate-compression`
Evaluates compressed text against original text using a specified metric.
**Usage:** `compact-memory dev evaluate-compression [OPTIONS] ORIGINAL_INPUT COMPRESSED_INPUT`
**Arguments:**
*   `ORIGINAL_INPUT`: Original text content, path to a text file, or '-' to read from stdin. (Required)
*   `COMPRESSED_INPUT`: Compressed text content, path to a text file, or '-' to read from stdin. (Required)
**Options:**
*   `--metric / -m TEXT`: ID of the validation metric to use (see 'list-metrics'). (Required)
*   `--metric-params JSON`: Metric parameters as a JSON string (e.g., '{"model_name": "bert-base-uncased"}'). *(Note: This option might be replaced by `--params-file` and `--metric-param` in future versions or was handled by `cli_utils.parse_complex_params` which is not directly exposed here).*
*   `--json`: Output evaluation scores in JSON format.

#### `compact-memory dev test-llm-prompt`
Tests a Language Model (LLM) prompt with specified context and query.
**Usage:** `compact-memory dev test-llm-prompt [OPTIONS]`
**Options:**
*   `--context / -c TEXT`: Context string for the LLM, path to a context file, or '-' to read from stdin. (Required)
*   `--query / -q TEXT`: User query to append to the context for the LLM. (Required)
*   `--model TEXT`: Model ID to use for the test (must be defined in LLM config) (default: "tiny-gpt2").
*   `--system-prompt / -s TEXT`: Optional system prompt to prepend to the main prompt.
*   `--max-new-tokens INTEGER`: Maximum number of new tokens the LLM should generate (default: 150).
*   `--output-response PATH`: File path to save the LLM's raw response. If unspecified, prints to console.
*   `--llm-config PATH`: Path to the LLM configuration YAML file (default: "llm_models_config.yaml").
*   `--api-key-env-var TEXT`: Environment variable name that holds the API key for the LLM provider (e.g., 'OPENAI_API_KEY').

#### `compact-memory dev evaluate-llm-response`
Evaluates an LLM's response against a reference answer using a specified metric.
**Usage:** `compact-memory dev evaluate-llm-response [OPTIONS] RESPONSE_INPUT REFERENCE_INPUT`
**Arguments:**
*   `RESPONSE_INPUT`: LLM's generated response text, path to a response file, or '-' to read from stdin. (Required)
*   `REFERENCE_INPUT`: Reference (ground truth) answer text or path to a reference file. (Required)
**Options:**
*   `--metric / -m TEXT`: ID of the validation metric to use (see 'list-metrics'). (Required)
*   `--metric-params JSON`: Metric parameters as a JSON string. *(See note for `evaluate-compression`)*.
*   `--json`: Output evaluation scores in JSON format.

#### `compact-memory dev download-embedding-model`
Downloads a specified SentenceTransformer embedding model from Hugging Face.
**Usage:** `compact-memory dev download-embedding-model [OPTIONS]`
**Options:**
*   `--model-name TEXT`: Name of the SentenceTransformer model to download (default: "all-MiniLM-L6-v2").

#### `compact-memory dev download-chat-model`
Downloads a specified causal Language Model (e.g., for chat) from Hugging Face.
**Usage:** `compact-memory dev download-chat-model [OPTIONS]`
**Options:**
*   `--model-name TEXT`: Name of the Hugging Face causal LM to download (default: "tiny-gpt2").

#### `compact-memory dev create-engine-package`
Creates a new compression engine extension package from a template.
**Usage:** `compact-memory dev create-engine-package [OPTIONS]`
**Options:**
*   `--name TEXT`: Name for the new engine package (e.g., `compact_memory_my_engine`). Used for directory and engine ID (default: "compact_memory_example_engine").
*   `--path PATH`: Directory where the engine package will be created. Defaults to a new directory named after the engine in the current location.

#### `compact-memory dev validate-engine-package`
Validates the structure and manifest of a compression engine extension package.
**Usage:** `compact-memory dev validate-engine-package [OPTIONS] PACKAGE_PATH`
**Arguments:**
*   `PACKAGE_PATH`: Path to the root directory of the engine package. (Required)

#### `compact-memory dev run-package-experiment`
Runs an experiment defined within a compression engine extension package.
**Usage:** `compact-memory dev run-package-experiment [OPTIONS] PACKAGE_PATH`
**Arguments:**
*   `PACKAGE_PATH`: Path to the root directory of the engine package. (Required)
**Options:**
*   `--experiment TEXT`: Name or relative path of the experiment configuration YAML file within the package's 'experiments' directory. If not specified, attempts to run a default experiment if defined in manifest.

#### `compact-memory dev run-hpo-script`
Executes a Python script, typically for Hyperparameter Optimization (HPO).
**Usage:** `compact-memory dev run-hpo-script [OPTIONS] SCRIPT_PATH`
**Arguments:**
*   `SCRIPT_PATH`: Path to the Python HPO script to execute. (Required)

#### `compact-memory dev inspect-trace`
Inspects a CompressionTrace JSON file, optionally filtering by step type.
**Usage:** `compact-memory dev inspect-trace [OPTIONS] TRACE_FILE`
**Arguments:**
*   `TRACE_FILE`: Path to the CompressionTrace JSON file. (Required)
**Options:**
*   `--type TEXT`: Filter trace steps by this 'type' string (e.g., 'chunking', 'llm_call').

---

*This reference should be updated as the CLI evolves.*
```
