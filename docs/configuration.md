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
    compact-memory compress --file my_document.txt
    compact-memory query "What was in my document?"
    ```

By effectively using the configuration layers, especially `compact-memory config set` for your global defaults, you can significantly simplify your interactions with the Compact Memory CLI. Remember that command-line options always override these defaults if you need to work with a different agent or setting temporarily.

## LLM Model Configurations (`llm_models_config.yaml`)

For more complex LLM configurations, especially when using multiple LLMs or different settings for various tasks, you can create an `llm_models_config.yaml` file. By default, Compact Memory looks for this file in the current working directory (your project's root).

This file allows you to define named configurations for different LLM providers and models. You can then refer to these configurations using the `--llm-config NAME` option in commands like `compact-memory compress` or `compact-memory engine init`.

**Example `llm_models_config.yaml`:**

```yaml
# llm_models_config.yaml example

my-local-summarizer:
  provider: local
  # model_path: /path/to/your/models/mistral-7b-instruct  # If your local provider uses model_path
  model_name: mistralai/Mistral-7B-Instruct-v0.1 # Or a Hugging Face model name
  # Other parameters specific to the local provider can be added here.

my-openai-gpt4:
  provider: openai
  model_name: gpt-4-turbo
  # api_key: sk-YOUR_OPENAI_KEY_HERE # Optional: can be stored here or in env variables.
  # max_tokens: 8000 # Example of a parameter an engine might use if passed this config.

my-openai-gpt3-5:
  provider: openai
  model_name: gpt-3.5-turbo

another-mock-config:
  provider: mock
  # Mock provider usually doesn't need more config, but you could add custom fields if your mock was extended.

# Configuration for a Gemini model
my-gemini-pro:
  provider: gemini
  model_name: gemini-pro
  # api_key: YOUR_GOOGLE_API_KEY # Optional, can use GOOGLE_API_KEY env var
```

**Key Fields:**

*   **`provider`**: (Mandatory) Specifies the type of LLM provider. Examples: `local`, `openai`, `gemini`, `mock`.
*   **`model_name`**: The identifier for the model used by the provider (e.g., `gpt-4-turbo` for OpenAI, `mistralai/Mistral-7B-Instruct-v0.1` for a Hugging Face model used with `local` provider).
*   **`model_path`**: Some local providers might use this to specify the direct path to model files. (Usage depends on the specific local provider's implementation).
*   **`api_key`**: API key for remote services like OpenAI or Gemini.
    *   **Security Note:** Storing API keys in plaintext YAML is convenient for personal use but is generally discouraged for security reasons, especially if the file is shared or version-controlled. It's often safer to use environment variables (e.g., `OPENAI_API_KEY`, `GOOGLE_API_KEY`) which the providers typically read automatically.
*   Other fields (like `max_tokens` in the example) can be added. These are not directly used by the LLM provider factory itself but can be part of the configuration dictionary passed to an engine if the engine is designed to interpret them.

When you use the `--llm-config NAME` CLI flag (e.g., `compact-memory compress ... --llm-config my-local-summarizer`), Compact Memory will load the settings from the corresponding block in `llm_models_config.yaml`.

This system allows you to manage multiple LLM setups cleanly and switch between them easily for different operations or projects.
```
