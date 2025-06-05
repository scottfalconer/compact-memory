# Configuring Compact Memory

Compact Memory offers a flexible configuration system to tailor its behavior to your needs and streamline your command-line usage. You can set default values for commonly used options so you don't have to type them every time.

## Configuration Layers and Precedence

Compact Memory resolves settings from the following sources, in order of highest to lowest precedence:

1.  **Command-Line Arguments:**
    Options provided directly with a command will always take top priority.
    *Example:* `compact-memory query "Hello" --model-id openai/gpt-4` will use `openai/gpt-4` for this query, regardless of other settings.

2.  **Environment Variables:**
    You can set specific environment variables to configure Compact Memory globally for your shell session or system.
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
    default_model_id: openai/gpt-3.5-turbo
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
    Once set, commands like `compact-memory ingest ...` or `compact-memory query ...` will use this path automatically unless you override it with the `--memory-path` option for a specific command.

*   `default_model_id`: Your preferred LLM for queries and other LLM-dependent operations.
    ```bash
    compact-memory config set default_model_id openai/gpt-4-turbo
    ```
    Commands like `compact-memory query` will use this model by default.

*   `default_engine_id`: Your preferred compression engine for summarization or queries (if applicable).
    ```bash
    compact-memory config set default_engine_id prototype
    ```
    Commands like `compact-memory compress` will use this strategy by default if you don't specify one with `--strategy`.

*(Note: While `log_file` and `verbose` can be set in config files manually, they are primarily controlled via CLI options for runtime flexibility. The `config set` command currently supports `compact_memory_path`, `default_model_id`, and `default_engine_id` as these are the most common global defaults users might want to persist.)*


**Example Workflow:**

1.  **Set your preferred memory path:**
    ```bash
    compact-memory config set compact_memory_path "~/compact_memories"
    ```
    *(The CLI will expand `~` to your home directory).*
2.  **Set your default LLM:**
    ```bash
    compact-memory config set default_model_id "openai/gpt-3.5-turbo"
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
    compact-memory ingest my_document.txt
    compact-memory query "What was in my document?"
    ```

By effectively using the configuration layers, especially `compact-memory config set` for your global defaults, you can significantly simplify your interactions with the Compact Memory CLI. Remember that command-line options always override these defaults if you need to work with a different agent or setting temporarily.
```
