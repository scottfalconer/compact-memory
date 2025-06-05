# Compact Memory

Compact Memory is a toolkit for compressing and managing text context for Large
Language Models (LLMs). It helps you keep important information accessible while
staying within tight token budgets.

For the complete project documentation, see [docs/README.md](docs/README.md).

The project offers:
*   **A ready-to-use toolkit** – Command-Line Interface and Python API for
    applying compression engines in your own pipelines.
*   **A flexible framework** – utilities for developing and testing new
    compression engines.

Traditional approaches such as standard Retrieval Augmented Generation (RAG) or
simply expanding context windows often fall short on complex tasks that require
dynamic learning and resource-conscious operation. Compact Memory addresses these
challenges by facilitating advanced memory compression techniques.

The benefits of using Compact Memory include:
-   Developing engines for evolving gist-based understanding, where memory consolidates and adapts over time.
-   Optimizing context for resource efficiency, managing token budgets to reduce API costs, lower latency, and enable the use of smaller or local LLMs.
-   Creating and utilizing learned compression and summarization techniques that can adapt to specific tasks or data types.
-   Implementing active memory management systems that simulate dynamic working memory for more coherent, long-running interactions.

## Who is Compact Memory For?

Compact Memory is built for those pushing the boundaries of LLM capabilities:

-   **Researchers:** Seeking a standardized environment to benchmark novel memory architectures, test hypotheses, and share reproducible findings on LLM memory engines.
-   **Developers:** Aiming to equip their LLM applications with more powerful, adaptive, and resource-efficient memory capabilities than off-the-shelf solutions provide.

## Using Compact Memory: The Toolkit

This section is for users who want to quickly leverage Compact Memory to compress text and optimize context for their LLM applications.

Compact Memory is designed for easy integration into your existing workflows.

### Installation

Get started by installing the `compact-memory` package:

```bash
pip install compact-memory
# Download recommended models for default engines and examples
python -m spacy download en_core_web_sm
compact-memory dev download-embedding-model
compact-memory dev download-chat-model
```
For more detailed installation options, see the [Installation](#installation) section.

### Quick Example

Compress a text file to 100 tokens using the default engine:

```bash
compact-memory compress --file my_document.txt --budget 100
```

See [USAGE.md](docs/USAGE.md) for more CLI examples and Python API usage.

<!-- Detailed usage examples have moved to USAGE.md -->

## Developing Engines: The Framework

For researchers and developers interested in creating novel context compression techniques, Compact Memory offers a comprehensive framework. It provides the necessary abstractions, tools, and testing utilities to design, build, and validate your own `CompressionEngine` implementations.

Key aspects of the framework include:

*   **`CompressionEngine` Base Class:** A clear abstract base class that defines the interface for all compression engines. Implement this to integrate your custom logic into the Compact Memory ecosystem.
*   **Validation Metrics:** A suite of metrics and the ability to define custom ones (`ValidationMetric`) to rigorously evaluate the performance and effectiveness of your CompressionEngines.
*   **Plugin Architecture:** A system for packaging and sharing your CompressionEngines, making them discoverable and usable by others.

To get started with building your own compression engines, please refer to our detailed guide:
*   **[Developing Compression Engines](docs/DEVELOPING_COMPRESSION_ENGINES.md)**

This guide will walk you through the process of creating a new CompressionEngine, from understanding the core components to testing and evaluation.


## Sharing and Discovering Engines

Compact Memory is designed to foster a community of innovation around LLM context compression. A key part of this is the ability to easily share engines you develop and discover engines created by others.

### Plugin System

Compact Memory features a plugin system that allows new compression engines to be discovered and loaded at runtime. If a Python package registers an engine using the `compact_memory.engines` entry point, or if a valid engine package directory is placed in a designated plugins path, Compact Memory will automatically make it available for use.

This makes it simple to extend the capabilities of Compact Memory without modifying the core codebase.

### Creating a Shareable Engine Package

To package your custom engine for sharing, Compact Memory provides a command-line tool to generate a template package structure. This includes the necessary metadata files and directory layout.

Use the `dev create-engine-package` command:

```bash
compact-memory dev create-engine-package --name compact_memory_my_engine
```

This will create a new directory (e.g., `compact_memory_my_engine/`) containing:
*   `engine.py`: A template for your engine code.
*   `engine_package.yaml`: A manifest file describing your engine.
*   `README.md`: Basic documentation for your package.
*   `requirements.txt`: For any specific dependencies your engine might have.

After populating these files with your engine's logic and details, it can be shared.

The recommended package name pattern for publishing on PyPI or GitHub is `compact_memory_<name>_engine`.

For comprehensive details on packaging and the plugin architecture, see:
*   **[Sharing Engines](docs/SHARING_ENGINES.md)**

### Finding, Installing, and Using Shared Engines

Shared engines can be distributed as standard Python packages (e.g., via PyPI) or as simple directory packages.

*   **Installation (Python Packages):** If an engine is distributed as a Python package, you can typically install it using pip:
    ```bash
    pip install some-compact-memory-engine-package
    ```
    Once installed in your Python environment, Compact Memory's plugin loader will automatically discover the engine if it correctly uses the entry point system.

*   **Installation (Directory Packages):** For engines distributed as a directory, you can place them in a location scanned by Compact Memory. (Refer to `docs/SHARING_ENGINES.md` for details on plugin paths like `$COMPACT_MEMORY_PLUGINS_PATH`).

*   **Using a Shared Engine:** Once installed and discovered, you can use a shared engine like any built-in engine by specifying its `engine_id` in the CLI or Python API:
    ```bash
    compact-memory compress --text "my text" --engine community_engine_id --budget 100
    ```
    ```python
    from CompressionEngine.core import get_compression_engine # Updated path
    engine = get_compression_engine("community_engine_id")() # Updated function and variable name
    # ... use the engine
    ```

You can list all available engines, including those from plugins, using:
```bash
compact-memory dev list-engines
```

## Navigate Compact Memory

This section guides you through different parts of the Compact Memory project, depending on your goals.

### Getting Started Quickly

For users who want to install Compact Memory and see it in action:

-   Follow the **[Installation](#installation)** instructions below.
-   Walk through the **[Core Workflow](#core-workflow)** examples to understand basic usage.

### Understanding Core Concepts

For users wanting to understand the foundational ideas behind Compact Memory:

-   **`CompressionEngine`**: An algorithm that takes input text and a token budget, and produces a compressed representation of that text suitable for an LLM prompt.
-   **`ValidationMetric`**: A method to evaluate the quality or utility of the compressed memory, often by assessing an LLM's performance on a task using that compressed memory.
-   For a deeper dive into concepts and architecture, see our main documentation portal in `docs/README.md` (or `docs/index.md`).
-   For conceptual background on memory engines, refer to `docs/PROJECT_VISION.md`.

### Glossary

* **CompressionEngine** – A pluggable algorithm that compresses input text into a shorter form while preserving meaning.

### Developing for Compact Memory

For contributors or those looking to build custom solutions on top of Compact Memory:

-   Learn about developing custom `CompressionEngine` implementations in `docs/DEVELOPING_COMPRESSION_ENGINES.md`.
-   Understand how to create new `ValidationMetric` functions by reading `docs/DEVELOPING_VALIDATION_METRICS.md`.
-   Review the overall system design in `docs/ARCHITECTURE.md`.

## Features

- Command-line interface for memory container management (`memory init`, `memory stats`, `memory validate`, `memory clear`), data processing (`ingest`, `query`, `compress`), configuration (`config set`, `config show`), and developer tools (`dev list-engines`, `dev evaluate-compression`, etc.).
- Global configuration options settable via CLI, environment variables, or config files.
- Pluggable memory compression engines.
 - Pluggable CompressionEngines.
- Pluggable embedding backends: random (default), OpenAI, or local sentence transformers.
- Chunks rendered using a canonical **WHO/WHAT/WHEN/WHERE/WHY** template before embedding.
- Runs smoothly in Colab; a notebook-based GUI is planned.
- Python API for decoding and summarizing prototypes.
- Interactive query interface via the `query` command.

## Memory Engines

Compact Memory supports various compression engines. You can list available engines, including those from plugins:
```bash
compact-memory dev list-engines
```

| ID (Examples)        | Description                                            |
| -------------------- | ------------------------------------------------------ |
| `prototype_engine`   | Prototype-based long-term memory with evolving summaries |
| `first_last_engine`  | Simple extractive: first and last N tokens/sentences   |
| `your_custom_plugin` | An engine you develop and load as a plugin            |
*(This table is illustrative; use `dev list-engines` for the current list)*

To use a specific engine, you can set it as a global default or specify it per command:
```bash
# Set default engine globally
compact-memory config set default_engine_id prototype_engine

# Use a specific engine for a compress command
compact-memory compress --text "My text..." --engine first_last_engine --budget 100
```

Plugins can add more engines. For example, the `rationale_episode` engine lives in the optional
`compact_memory_rationale_episode_engine` package. Install it with
`pip install compact_memory_rationale_episode_engine` to enable it.
You can then set it via `compact-memory config set default_engine_id rationale_episode_engine`.
Note: The old method of enabling engines via `compact_memory_config.yaml` directly is being phased out in favor of the `config set` command and plugin system.

### Using Experimental Engines

Experimental engines live under `CompressionEngine.contrib`.
The CLI registers them automatically, so commands like `--engine first_last_engine` work out of the box.
When using the Python API directly, call `enable_all_experimental_engines()` to register them:

```python
from CompressionEngine.contrib import ( # Updated path
    ChainedEngine, # Updated class name
    enable_all_experimental_engines, # Updated function name
)

enable_all_experimental_engines() # Updated function name
```

Once registered, these engines behave like any other:

```bash
compact-memory compress --file text.txt --engine chained_engine --budget 200
```

Contrib engines are experimental and may change without notice.

## Why Compact Memory?

Compact Memory offers unique advantages for advancing LLM memory capabilities:

*   **For Researchers:**
    *   Provides a standardized platform to benchmark and compare novel memory compression algorithms.
    *   Facilitates reproducible experiments and sharing of findings within the community.
*   **For Developers:**
    *   Enables the integration of sophisticated, adaptive memory into LLM applications.
    *   Helps optimize context windows, reduce token usage, and improve performance with large information loads.
*   **For the Broader Community:**
    *   Fosters collaboration and discovery of new techniques for efficient LLM memory management.

Key benefits include:
*   **Evolving Understanding:** Go beyond standard Retrieval Augmented Generation (RAG) by creating systems where memory consolidates, adapts, and forms conceptual "gists" from information over time.
*   **Resource Optimization:** Optimize LLM interactions in resource-constrained settings by efficiently managing token budgets, potentially reducing API costs, latency, and enabling the use of smaller or local models.
*   **Learned Compression:** Facilitate research and implementation of engines that can be trained or adapt to optimally compress information for specific tasks or data types.

## Installation

This project requires **Python 3.11+**.

1.  **Install Dependencies:** Install the lightweight requirements listed in
   `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
   Optional features such as spaCy sentence segmentation or local models
   require extra dependencies. Install them with pip extras, e.g.:
   ```bash
   pip install .[spacy]        # for robust sentence segmentation
   pip install .[embedding]    # to enable embedding pipelines
   pip install .[local]        # for local Transformers models
   pip install .[gemini]       # Google Gemini provider
   pip install .[metrics]      # Hugging Face evaluation metrics
   ```
   To run the full test suite with all heavy packages, install additional
   dependencies:
   ```bash
   pip install torch sentence-transformers transformers spacy \
       google-generativeai evaluate
   ```
Note: The repository also includes `.codex/setup.sh`, which is intended solely for
the Codex environment and is not needed for general users.

2.  **Install `compact-memory`:**
    This makes the `compact-memory` CLI tool available. You have two main options:
    *   **Editable Install (Recommended for Development):** This installs the package in a way that changes to the source code are immediately reflected.
        ```bash
        pip install -e .
        ```
    *   **Standard Install:**
        ```bash
        pip install .
        ```

3.  **Download Models for Examples and Testing (Optional but Recommended):**
    These models are used by some of the example strategies and for testing LLM interactions with compressed memory.
    ```bash
    # Fetch the "all-MiniLM-L6-v2" model for embedding (used by default LTM components)
    compact-memory dev download-embedding-model --model-name all-MiniLM-L6-v2
    # Fetch a default chat model for the 'query' command and LLM-based validation
    compact-memory dev download-chat-model --model-name tiny-gpt2
    ```
    Note: Specific `CompressionEngine` implementations might have other model dependencies not covered here. Always check the documentation for the engine you intend to use.

4.  **Set API Keys for Cloud Providers (Optional):**
    If you plan to use OpenAI or Gemini models with `compact-memory`, export your API keys as environment variables:
    ```bash
    export OPENAI_API_KEY="sk-..."
    export GEMINI_API_KEY="..."
    ```

Run `compact-memory --help` to see available commands and verify installation.

You can also set a default location for the on-disk memory store and other global settings. See the "Configuration" section below.

## Configuration

Compact Memory uses a hierarchical configuration system:
1.  **Command-line arguments:** Highest precedence (e.g., `compact-memory --memory-path ./my_memory ingest ...`).
2.  **Environment variables:** (e.g., `COMPACT_MEMORY_PATH`, `COMPACT_MEMORY_DEFAULT_MODEL_ID`, `COMPACT_MEMORY_DEFAULT_ENGINE_ID`).
3.  **Local project config:** `.gmconfig.yaml` in the current directory.
4.  **User global config:** `~/.config/compact_memory/config.yaml`.
5.  **Hardcoded defaults.**

You can manage the user global configuration using the `config` commands:
-   `compact-memory config show`: Displays the current effective configuration and where each value originates.
-   `compact-memory config set <KEY> <VALUE>`: Sets a key in the user global config file.

For example, to set your default memory path globally:
```bash
# Set a default directory for your stored memories
compact-memory config set compact_memory_path ~/my_compact_memories
# Subsequent commands will use this path unless overridden by --memory-path or an environment variable.
```
Or, set it using an environment variable for the current session:
```bash
export COMPACT_MEMORY_PATH=~/my_compact_memories
```

## Quick Start / Core Workflow

The `compact-memory` Command-Line Interface (CLI) is your primary tool for managing memory containers, ingesting data, querying, and summarizing.

**1. Initialize a Memory Container:**
First, create a new memory container. This directory will store the container's data.
```bash
compact-memory memory init ./my_memory --model-name sentence-transformers/all-MiniLM-L6-v2
```
This creates a memory container at `./my_memory`. The specified embedding model will be downloaded if not already present.

**2. Configure Memory Path (Optional but Recommended for Convenience):**
To avoid specifying `--memory-path ./my_memory` for every command that interacts with this container, you can set it globally for your user or for the current terminal session.

*   **Set globally (user config):**
    ```bash
    compact-memory config set compact_memory_path ./my_memory
    ```
    Now, `compact-memory` commands like `ingest` and `query` will default to using `./my_memory`.
*   **Set for current session (environment variable):**
    ```bash
    export COMPACT_MEMORY_PATH=$(pwd)/my_memory
    ```

**3. Ingest Data:**
Add information to the memory container.
```bash
# If compact_memory_path is set (globally or via env var):
compact-memory ingest path/to/your_document.txt
compact-memory ingest path/to/your_data_directory/

# Or, specify the memory path directly for a specific command:
compact-memory --memory-path ./my_memory ingest path/to/your_document.txt
```

**4. Query the Container:**
Ask questions based on the ingested information. The container uses its configured default model and engine unless overridden.
```bash
# If compact_memory_path is set:
compact-memory query "What was mentioned about project X?"

# Or, specify the memory path directly:
compact-memory --memory-path ./my_memory query "What was mentioned about project X?"

# You can also override the default model or engine for a specific query:
compact-memory query "Summarize recent findings on AI ethics" --model-id openai/gpt-4-turbo --engine prototype_engine
```

**5. Compress Text (Standalone Utility):**
Compress text using a specific engine without necessarily interacting with a container's stored memory. This is useful for quick text compression tasks.
```bash
compact-memory compress --text "This is a very long piece of text that needs to be shorter." --engine first_last_engine --budget 50
compact-memory compress --file path/to/another_document.txt -e prototype_engine -b 200 -o compressed_summary.txt
```
Note: `-s` was assumed to be short for `--strategy`, changed to `-e` for `--engine`. If `-s` had a different meaning, this would need adjustment.

**Developer Tools & Evaluation:**
Compact Memory also includes tools for developers and researchers, such as evaluating compression engines or testing LLM prompts. These are typically found under the `dev` subcommand group.

For example, to evaluate compression quality:
```bash
compact-memory dev evaluate-compression original.txt compressed_version.txt --metric compression_ratio
```
To list available engines (including plugins):
```bash
compact-memory dev list-engines
```

### Running Tests
To ensure Compact Memory is functioning correctly, especially after making changes or setting up the environment:
Install development dependencies (if not already done during setup):
```bash
pip install -r requirements.txt  # Ensure test dependencies are included
pytest
```


## Documentation

For more detailed information on Compact Memory's architecture, development guides, and advanced topics, please refer to the `docs/` directory.

-   **Main Documentation Portal:** `docs/README.md` (or `docs/index.md`) serves as a Table of Contents and entry point for deeper documentation.
-   **Architecture Deep Dive:** Understand the overall system design in `docs/ARCHITECTURE.md`.
-   **Developing Compression Engines:** Learn how to create your own engines in `docs/DEVELOPING_COMPRESSION_ENGINES.md`.
-   **Developing Validation Metrics:** Find guidance on building custom metrics in `docs/DEVELOPING_VALIDATION_METRICS.md`.
-   **Plugins:** Learn how to install or develop engine plugins in `docs/SHARING_ENGINES.md`.
-   **Conceptual Guides:** Explore the ideas behind memory engines in `docs/PROJECT_VISION.md`.

## Designing Compression Engines

Compact Memory is designed to support a wide variety of `CompressionEngine` implementations. For detailed guidance on creating your own, including best practices for splitting documents into meaningful chunks (e.g., belief-sized ideas) and techniques for updating memory (like centroid updates), please see:

-   `docs/COMPRESSION_ENGINES.md`
-   `docs/DEVELOPING_COMPRESSION_ENGINES.md`

The `AgenticChunker` is an example of an advanced chunking mechanism. You can enable it during memory initialization (e.g., `compact-memory memory init ./my_memory --chunker agentic`) or programmatically within your custom engine (e.g., `agent.chunker = AgenticChunker()`).

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines on setting up a development environment, coding style, and how to run tests before submitting pull requests.

## Query Tips

Effective querying is crucial when working with compressed memory. For tips on how to shape your search queries to best retrieve information, especially when your notes follow a structured format (like WHO/WHAT/WHEN), refer to:

-   `docs/QUERY_TIPS.md`

This document explains techniques such as templating your questions to align with the structure of your memory store, thereby biasing retrieval towards more relevant results.

## Preprocessing & Cleanup

Compact Memory no longer performs built-in line filtering or heuristic cleanup when ingesting text. You can supply any callable that transforms raw strings before they are chunked or embedded:

```python
def remove_blank_lines(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if line.strip())

# Example of applying a preprocessing function before engine processing:
# (Conceptual: Actual API for applying this would depend on the specific workflow)
# processed_text = remove_blank_lines(raw_text)
# compressed_output = compression_engine.compress(processed_text, ...)
```

This hook enables custom regex cleanup, spaCy pipelines or LLM-powered summarization prior to compression.

## Architecture and Storage

Compact Memory features a modular architecture that allows for flexible extension and adaptation. The core interfaces for `CompressionEngine` and `ValidationMetric` are designed to be pluggable, enabling diverse implementations.

Persistent storage is primarily relevant for `CompressionEngine` implementations that maintain a stateful long-term memory (e.g., engines based on prototypes or evolving summaries). Many engines, however, can be stateless, processing input text without relying on persistent memory.

For a detailed architectural overview and discussion of storage options, please see:
-   `docs/ARCHITECTURE.md`

This document provides a high-level view of the components and their interactions.
