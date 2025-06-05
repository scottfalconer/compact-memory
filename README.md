# Compact Memory

Compact Memory is a toolkit for compressing and managing text context for Large
Language Models (LLMs). It helps you keep important information accessible while
staying within tight token budgets.

The project offers:
*   **A ready-to-use toolkit** – Command-Line Interface and Python API for
    applying compression strategies in your own pipelines.
*   **A flexible framework** – utilities for developing and testing new
    compression strategies.

Traditional approaches such as standard Retrieval Augmented Generation (RAG) or
simply expanding context windows often fall short on complex tasks that require
dynamic learning and resource-conscious operation. Compact Memory addresses these
challenges by facilitating advanced memory compression techniques.

The benefits of using Compact Memory include:
-   Developing strategies for evolving gist-based understanding, where memory consolidates and adapts over time.
-   Optimizing context for resource efficiency, managing token budgets to reduce API costs, lower latency, and enable the use of smaller or local LLMs.
-   Creating and utilizing learned compression and summarization techniques that can adapt to specific tasks or data types.
-   Implementing active memory management systems that simulate dynamic working memory for more coherent, long-running interactions.

## Who is Compact Memory For?

Compact Memory is built for those pushing the boundaries of LLM capabilities:

-   **Researchers:** Seeking a standardized environment to benchmark novel memory architectures, test hypotheses, and share reproducible findings on LLM memory strategies.
-   **Developers:** Aiming to equip their LLM applications with more powerful, adaptive, and resource-efficient memory capabilities than off-the-shelf solutions provide.

## Using Compact Memory: The Toolkit

This section is for users who want to quickly leverage Compact Memory to compress text and optimize context for their LLM applications.

Compact Memory is designed for easy integration into your existing workflows.

### Installation

Get started by installing the `compact-memory` package:

```bash
pip install compact-memory
# Download recommended models for default strategies and examples
python -m spacy download en_core_web_sm
compact-memory dev download-embedding-model
compact-memory dev download-chat-model
```
For more detailed installation options, see the [Installation](#installation) section.

### Quick Example

Compress a text file to 100 tokens using the default strategy:

```bash
compact-memory compress --file my_document.txt --budget 100
```

See [USAGE.md](USAGE.md) for more CLI examples and Python API usage.

<!-- Detailed usage examples have moved to USAGE.md -->

## Developing Strategies: The Framework

For researchers and developers interested in creating novel context compression techniques, Compact Memory offers a comprehensive framework. It provides the necessary abstractions, tools, and testing utilities to design, build, and validate your own `CompressionStrategy` implementations.

Key aspects of the framework include:

*   **`CompressionStrategy` Base Class:** A clear abstract base class that defines the interface for all compression strategies. Implement this to integrate your custom logic into the Compact Memory ecosystem.
*   **Validation Metrics:** A suite of metrics and the ability to define custom ones (`ValidationMetric`) to rigorously evaluate the performance and effectiveness of your strategies.
*   **Plugin Architecture:** A system for packaging and sharing your strategies, making them discoverable and usable by others.

To get started with building your own compression strategies, please refer to our detailed guide:
*   **[Developing Compression Strategies](docs/DEVELOPING_COMPRESSION_STRATEGIES.md)**

This guide will walk you through the process of creating a new strategy, from understanding the core components to testing and evaluation.

## Sharing and Discovering Strategies

Compact Memory is designed to foster a community of innovation around LLM context compression. A key part of this is the ability to easily share strategies you develop and discover strategies created by others.

### Plugin System

Compact Memory features a plugin system that allows new compression strategies to be discovered and loaded at runtime. If a Python package registers a strategy using the `compact_memory.strategies` entry point, or if a valid strategy package directory is placed in a designated plugins path, Compact Memory will automatically make it available for use.

This makes it simple to extend the capabilities of Compact Memory without modifying the core codebase.

### Creating a Shareable Strategy Package

To package your custom strategy for sharing, Compact Memory provides a command-line tool to generate a template package structure. This includes the necessary metadata files and directory layout.

Use the `dev create-strategy-package` command:

```bash
compact-memory dev create-strategy-package --name YourStrategyName
```

This will create a new directory (e.g., `YourStrategyName/`) containing:
*   `strategy.py`: A template for your strategy code.
*   `strategy_package.yaml`: A manifest file describing your strategy.
*   `README.md`: Basic documentation for your package.
*   `requirements.txt`: For any specific dependencies your strategy might have.

After populating these files with your strategy's logic and details, it can be shared.

For comprehensive details on packaging and the plugin architecture, see:
*   **[Sharing Strategies](docs/SHARING_STRATEGIES.md)**

### Finding, Installing, and Using Shared Strategies

Shared strategies can be distributed as standard Python packages (e.g., via PyPI) or as simple directory packages.

*   **Installation (Python Packages):** If a strategy is distributed as a Python package, you can typically install it using pip:
    ```bash
    pip install some-compact-memory-strategy-package
    ```
    Once installed in your Python environment, Compact Memory's plugin loader will automatically discover the strategy if it correctly uses the entry point system.

*   **Installation (Directory Packages):** For strategies distributed as a directory, you can place them in a location scanned by Compact Memory. (Refer to `docs/SHARING_STRATEGIES.md` for details on plugin paths like `$COMPACT_MEMORY_PLUGINS_PATH`).

*   **Using a Shared Strategy:** Once installed and discovered, you can use a shared strategy like any built-in strategy by specifying its `strategy_id` in the CLI or Python API:
    ```bash
    compact-memory compress --text "my text" --strategy community_strategy_id --budget 100
    ```
    ```python
    from compact_memory.compression import get_compression_strategy
    strategy = get_compression_strategy("community_strategy_id")()
    # ... use the strategy
    ```

You can list all available strategies, including those from plugins, using:
```bash
compact-memory dev list-strategies
```

## Navigate Compact Memory

This section guides you through different parts of the Compact Memory project, depending on your goals.

### Getting Started Quickly

For users who want to install Compact Memory and see it in action:

-   Follow the **[Installation](#installation)** instructions below.
-   Walk through the **[Core Workflow](#core-workflow)** examples to understand basic usage.

### Understanding Core Concepts

For users wanting to understand the foundational ideas behind Compact Memory:

-   **`CompressionStrategy`**: An algorithm that takes input text and a token budget, and produces a compressed representation of that text suitable for an LLM prompt.
-   **`ValidationMetric`**: A method to evaluate the quality or utility of the compressed memory, often by assessing an LLM's performance on a task using that compressed memory.
-   For a deeper dive into concepts and architecture, see our main documentation portal in `docs/README.md` (or `docs/index.md`).
-   For conceptual background on memory strategies, refer to `docs/PROJECT_VISION.md`.

### Developing for Compact Memory

For contributors or those looking to build custom solutions on top of Compact Memory:

-   Learn about developing custom `CompressionStrategy` implementations in `docs/DEVELOPING_COMPRESSION_STRATEGIES.md`.
-   Understand how to create new `ValidationMetric` functions by reading `docs/DEVELOPING_VALIDATION_METRICS.md`.
-   Review the overall system design in `docs/ARCHITECTURE.md`.

## Features

- Command-line interface for agent management (`agent init`, `agent stats`, `agent validate`, `agent clear`), data processing (`ingest`, `query`, `compress`), configuration (`config set`, `config show`), and developer tools (`dev list-strategies`, `dev evaluate-compression`, etc.).
- Global configuration options settable via CLI, environment variables, or config files.
- Pluggable memory compression strategies.
- Pluggable embedding backends: random (default), OpenAI, or local sentence transformers.
- Chunks rendered using a canonical **WHO/WHAT/WHEN/WHERE/WHY** template before embedding.
- Runs smoothly in Colab; a notebook-based GUI is planned.
- Python API for decoding and summarizing prototypes.
- Interactive query interface via the `query` command.

## Memory Strategies

Compact Memory supports various compression strategies. You can list available strategies, including those from plugins:
```bash
compact-memory dev list-strategies
```

| ID (Examples)       | Description                                           |
| ------------------- | ----------------------------------------------------- |
| `prototype`         | Prototype-based long-term memory with evolving summaries |
| `first_last`        | Simple extractive: first and last N tokens/sentences  |
| `your_custom_plugin`| A strategy you develop and load as a plugin          |
*(This table is illustrative; use `dev list-strategies` for the current list)*

To use a specific strategy, you can set it as a global default or specify it per command:
```bash
# Set default strategy globally
compact-memory config set default_strategy_id prototype

# Use a specific strategy for a compress command
compact-memory compress --text "My text..." --strategy first_last --budget 100
```

Plugins can add more strategies. For example, the `rationale_episode` strategy lives in the optional
`compact_memory_rationale_episode_strategy` package. Install it with
`pip install compact_memory_rationale_episode_strategy` to enable it.
You can then set it via `compact-memory config set default_strategy_id rationale_episode`.
Note: The old method of enabling strategies via `compact_memory_config.yaml` directly is being phased out in favor of the `config set` command and plugin system.

### Using Experimental Strategies

Experimental strategies live under ``compact_memory.strategies.experimental``.
The CLI registers them automatically, so commands like ``--strategy first_last`` work out of the box.
When using the Python API directly, call ``enable_all_experimental_strategies()`` to register them:

```python
from compact_memory.strategies.experimental import (
    ChainedStrategy,
    enable_all_experimental_strategies,
)

enable_all_experimental_strategies()
```

Once registered, these strategies behave like any other:

```bash
compact-memory compress --file text.txt --strategy chained --budget 200
```

Contrib strategies are experimental and may change without notice.

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
*   **Learned Compression:** Facilitate research and implementation of strategies that can be trained or adapt to optimally compress information for specific tasks or data types.

## Installation

This project requires **Python 3.11+**.

1.  **Install Core Dependencies:** Use the provided `setup.sh` for a fast install.
   ```bash
   bash setup.sh           # installs the minimal set of packages
   # Download the spaCy model for sentence segmentation (used by some strategies/chunkers)
   python -m spacy download en_core_web_sm
   ```
   If you need to run the full test suite (which depends on PyTorch and other
   heavy packages), run:
   ```bash
   FULL_INSTALL=1 bash setup.sh
   ```
Note: The repository also includes `.codex/setup.sh`, which is intended solely for the Codex environment and is not needed for general users.

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
    Note: Specific `CompressionStrategy` implementations might have other model dependencies not covered here. Always check the documentation for the strategy you intend to use.

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
1.  **Command-line arguments:** Highest precedence (e.g., `compact-memory --memory-path ./my_agent ingest ...`).
2.  **Environment variables:** (e.g., `COMPACT_MEMORY_PATH`, `COMPACT_MEMORY_DEFAULT_MODEL_ID`, `COMPACT_MEMORY_DEFAULT_STRATEGY_ID`).
3.  **Local project config:** `.gmconfig.yaml` in the current directory.
4.  **User global config:** `~/.config/compact_memory/config.yaml`.
5.  **Hardcoded defaults.**

You can manage the user global configuration using the `config` commands:
-   `compact-memory config show`: Displays the current effective configuration and where each value originates.
-   `compact-memory config set <KEY> <VALUE>`: Sets a key in the user global config file.

For example, to set your default memory path globally:
```bash
compact-memory config set compact_memory_path ~/my_gist_memories
# Subsequent commands will use this path unless overridden by --memory-path or an environment variable.
```
Or, set it using an environment variable for the current session:
```bash
export COMPACT_MEMORY_PATH=~/my_gist_memories
```

## Quick Start / Core Workflow

The `compact-memory` Command-Line Interface (CLI) is your primary tool for managing memory agents, ingesting data, querying, and summarizing.

**1. Initialize an Agent:**
First, create a new memory agent. This directory will store the agent's data.
```bash
compact-memory agent init ./my_agent --model-name sentence-transformers/all-MiniLM-L6-v2
```
This creates an agent at `./my_agent`. The specified embedding model will be downloaded if not already present.

**2. Configure Memory Path (Optional but Recommended for Convenience):**
To avoid specifying `--memory-path ./my_agent` for every command that interacts with this agent, you can set it globally for your user or for the current terminal session.

*   **Set globally (user config):**
    ```bash
    compact-memory config set compact_memory_path ./my_agent
    ```
    Now, `compact-memory` commands like `ingest` and `query` will default to using `./my_agent`.
*   **Set for current session (environment variable):**
    ```bash
    export COMPACT_MEMORY_PATH=$(pwd)/my_agent
    ```

**3. Ingest Data:**
Add information to the agent's memory.
```bash
# If compact_memory_path is set (globally or via env var):
compact-memory ingest path/to/your_document.txt
compact-memory ingest path/to/your_data_directory/

# Or, specify the memory path directly for a specific command:
compact-memory --memory-path ./my_agent ingest path/to/your_document.txt
```

**4. Query the Agent:**
Ask questions based on the ingested information. The agent uses its configured default model and strategy unless overridden.
```bash
# If compact_memory_path is set:
compact-memory query "What was mentioned about project X?"

# Or, specify the memory path directly:
compact-memory --memory-path ./my_agent query "What was mentioned about project X?"

# You can also override the default model or strategy for a specific query:
compact-memory query "Summarize recent findings on AI ethics" --model-id openai/gpt-4-turbo --strategy prototype
```

**5. Compress Text (Standalone Utility):**
Compress text using a specific strategy without necessarily interacting with an agent's stored memory. This is useful for quick text compression tasks.
```bash
compact-memory compress --text "This is a very long piece of text that needs to be shorter." --strategy first_last --budget 50
compact-memory compress --file path/to/another_document.txt -s prototype -b 200 -o compressed_summary.txt
```

**Developer Tools & Evaluation:**
Compact Memory also includes tools for developers and researchers, such as evaluating compression strategies or testing LLM prompts. These are typically found under the `dev` subcommand group.

For example, to evaluate compression quality:
```bash
compact-memory dev evaluate-compression original.txt compressed_version.txt --metric compression_ratio
```
To list available strategies (including plugins):
```bash
compact-memory dev list-strategies
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
-   **Developing Compression Strategies:** Learn how to create your own strategies in `docs/DEVELOPING_COMPRESSION_STRATEGIES.md`.
-   **Developing Validation Metrics:** Find guidance on building custom metrics in `docs/DEVELOPING_VALIDATION_METRICS.md`.
-   **Plugins:** Learn how to install or develop strategy plugins in `docs/SHARING_STRATEGIES.md`.
-   **Conceptual Guides:** Explore the ideas behind memory strategies in `docs/PROJECT_VISION.md`.

## Designing Compression Strategies

Compact Memory is designed to support a wide variety of `CompressionStrategy` implementations. For detailed guidance on creating your own, including best practices for splitting documents into meaningful chunks (e.g., belief-sized ideas) and techniques for updating memory (like centroid updates), please see:

-   `docs/COMPRESSION_STRATEGIES.md`
-   `docs/DEVELOPING_COMPRESSION_STRATEGIES.md`

The `AgenticChunker` is an example of an advanced chunking mechanism. You can enable it during agent initialization (e.g., `compact-memory agent init ./my_memory --chunker agentic`) or programmatically within your custom strategy (e.g., `agent.chunker = AgenticChunker()`).

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on setting up a development environment, coding style, and how to run tests before submitting pull requests.

## Query Tips

Effective querying is crucial when working with compressed memory. For tips on how to shape your search queries to best retrieve information, especially when your notes follow a structured format (like WHO/WHAT/WHEN), refer to:

-   `docs/QUERY_TIPS.md`

This document explains techniques such as templating your questions to align with the structure of your memory store, thereby biasing retrieval towards more relevant results.

## Preprocessing & Cleanup

Compact Memory no longer performs built-in line filtering or heuristic cleanup when ingesting text. You can supply any callable that transforms raw strings before they are chunked or embedded:

```python
def remove_blank_lines(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if line.strip())

agent = Agent(store, preprocess_fn=remove_blank_lines)
```

This hook enables custom regex cleanup, spaCy pipelines or LLM-powered summarization prior to compression.

## Architecture and Storage

Compact Memory features a modular architecture that allows for flexible extension and adaptation. The core interfaces for `CompressionStrategy` and `ValidationMetric` are designed to be pluggable, enabling diverse implementations.

Persistent storage is primarily relevant for `CompressionStrategy` implementations that maintain a stateful long-term memory (e.g., strategies based on prototypes or evolving summaries). Many strategies, however, can be stateless, processing input text without relying on persistent memory.

For a detailed architectural overview and discussion of storage options, please see:
-   `docs/ARCHITECTURE.md`

This document provides a high-level view of the components and their interactions.
