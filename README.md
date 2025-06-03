# Gist Memory

Gist Memory: An Experimentation Platform for Advanced LLM Memory Strategies.

Large Language Models (LLMs) are transforming how we interact with information, yet their ability to maintain long-term, coherent, and efficient memory remains a significant challenge. Standard approaches like Retrieval Augmented Generation (RAG) and expanding context windows offer partial solutions, but often fall short for complex tasks requiring dynamic learning, nuanced understanding, and resource-conscious operation.

Gist Memory is an open-source platform dedicated to pioneering the next generation of LLM memory. It provides a robust framework for researchers and developers to design, rigorously test, and validate sophisticated `CompressionStrategy` implementations. Instead of just retrieving static data, Gist Memory facilitates the exploration of advanced techniques.

The benefits include:
- Evolving gist-based understanding: Strategies where memory consolidates, adapts, and forms conceptual "gists" from information over time.
- Resource-efficient context optimization: Methods to maximize information density within token budgets, critical for managing API costs, latency, and enabling smaller or local LLMs.
- Learned compression and summarization: Techniques that can be trained or adapt to optimally compress information for specific tasks or data types.
- Active memory management: Systems that simulate dynamic working memory, managing recency, relevance, and trace strength for coherent, long-running interactions.

## Who is Gist Memory For?

Gist Memory is built for those pushing the boundaries of LLM capabilities:

-   **Researchers:** Seeking a standardized environment to benchmark novel memory architectures, test hypotheses, and share reproducible findings on LLM memory strategies.
-   **Developers:** Aiming to equip their LLM applications with more powerful, adaptive, and resource-efficient memory capabilities than off-the-shelf solutions provide.

## Navigate Gist Memory

This section guides you through different parts of the Gist Memory project, depending on your goals.

### Getting Started Quickly

For users who want to install Gist Memory and see it in action:

-   Follow the **[Installation](#installation)** instructions below.
-   Walk through the **[Core Workflow](#core-workflow)** examples to understand basic usage.
-   Run the onboarding demo: `python examples/onboarding_demo.py` for a quick demonstration of the experimentation capabilities.

### Understanding Core Concepts

For users wanting to understand the foundational ideas behind Gist Memory:

-   **`CompressionStrategy`**: An algorithm that takes input text and a token budget, and produces a compressed representation of that text suitable for an LLM prompt.
-   **`ValidationMetric`**: A method to evaluate the quality or utility of the compressed memory, often by assessing an LLM's performance on a task using that compressed memory.
-   **Experimentation Framework**: The tools and processes within Gist Memory for systematically running tests with different strategies, datasets, and metrics.
-   For a deeper dive into concepts and architecture, see our main documentation portal in `docs/README.md` (or `docs/index.md`).
-   For conceptual background on memory strategies, refer to `AGENTS.md`.

### Developing for Gist Memory

For contributors or those looking to build custom solutions on top of Gist Memory:

-   Learn about developing custom `CompressionStrategy` implementations in `docs/DEVELOPING_COMPRESSION_STRATEGIES.md`.
-   Understand how to create new `ValidationMetric` functions by reading `docs/DEVELOPING_VALIDATION_METRICS.md`.
-   Review the overall system design in `docs/ARCHITECTURE.md`.

## Features

- Command-line interface for agent management (`agent init`, `agent stats`, `agent validate`, `agent clear`), data processing (`ingest`, `query`, `summarize`), configuration (`config set`, `config show`), and developer tools (`dev list-strategies`, `dev evaluate-compression`, etc.).
- Global configuration options settable via CLI, environment variables, or config files.
- Lightweight JSON/NPY backend for prototypes and memories with optional Chroma vector store for scale (`pip install "gist-memory[chroma]"`).
- Pluggable memory compression strategies.
- Pluggable embedding backends: random (default), OpenAI, or local sentence transformers.
- Chunks rendered using a canonical **WHO/WHAT/WHEN/WHERE/WHY** template before embedding.
- Runs smoothly in Colab; a notebook-based GUI is planned.
- Python API for decoding and summarizing prototypes.
- Interactive query interface via the `query` command.
- Debug logging with `--log-file` and conflict heuristics written to `conflicts.jsonl` for HITL review.

## Memory Strategies

Gist Memory supports various compression strategies. You can list available strategies, including those from plugins:
```bash
gist-memory dev list-strategies
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
gist-memory config set default_strategy_id prototype

# Use a specific strategy for a summarize command
gist-memory summarize "My text..." --strategy first_last --budget 100
```

Plugins can add more strategies. For example, the `rationale_episode` strategy lives in the optional
`gist_memory_rationale_episode_strategy` package. Install it with
`pip install gist_memory_rationale_episode_strategy` to enable it.
You can then set it via `gist-memory config set default_strategy_id rationale_episode`.
Note: The old method of enabling strategies via `gist_memory_config.yaml` directly is being phased out in favor of the `config set` command and plugin system.

## Why Gist Memory?

Gist Memory offers unique advantages for advancing LLM memory capabilities:

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

0.  **Run `setup.sh` (Optional for Offline-Friendly Environment):**
    To prepare an offline-friendly development environment quickly, you can execute the provided script while you have internet access. This script pre-downloads many dependencies.
    ```bash
    ./setup.sh
    ```
    This step is optional; you can install dependencies directly if you prefer.

1.  **Install Core Dependencies:**
    ```bash
    pip install -r requirements.txt
    # Download the spaCy model for sentence segmentation (used by some strategies/chunkers)
    python -m spacy download en_core_web_sm
    ```

2.  **Install `gist-memory`:**
    This makes the `gist-memory` CLI tool available. You have two main options:
    *   **Editable Install (Recommended for Development):** This installs the package in a way that changes to the source code are immediately reflected.
        ```bash
        pip install -e .
        ```
    *   **Standard Install:**
        ```bash
        pip install .
        ```
    *   **Offline Note:** If working in an offline environment after fetching the repository and dependencies (e.g., via `setup.sh`), you might need the `--no-build-isolation` flag with `pip install -e .` or `pip install .`. Alternatively, you can add the repository root to your `PYTHONPATH`:
        ```bash
        export PYTHONPATH="$(pwd):$PYTHONPATH"
        ```

3.  **Download Models for Examples and Testing (Optional but Recommended):**
    These models are used by some of the example strategies and for testing LLM interactions with compressed memory.
    ```bash
    # Fetch the "all-MiniLM-L6-v2" model for embedding (used by default LTM components)
    gist-memory dev download-embedding-model --model-name all-MiniLM-L6-v2
    # Fetch a default chat model for the 'query' command and LLM-based validation
    gist-memory dev download-chat-model --model-name tiny-gpt2
    ```
    Note: Specific `CompressionStrategy` implementations might have other model dependencies not covered here. Always check the documentation for the strategy you intend to use.
    To run completely offline after all downloads, set:
    ```
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    ```

4.  **Set API Keys for Cloud Providers (Optional):**
    If you plan to use OpenAI or Gemini models with `gist-memory`, export your API keys as environment variables:
    ```bash
    export OPENAI_API_KEY="sk-..."
    export GEMINI_API_KEY="..."
    ```

Run `gist-memory --help` to see available commands and verify installation.

You can also set a default location for the on-disk memory store and other global settings. See the "Configuration" section below.

## Configuration

Gist Memory uses a hierarchical configuration system:
1.  **Command-line arguments:** Highest precedence (e.g., `gist-memory --memory-path ./my_agent ingest ...`).
2.  **Environment variables:** (e.g., `GIST_MEMORY_PATH`, `GIST_MEMORY_DEFAULT_MODEL_ID`, `GIST_MEMORY_DEFAULT_STRATEGY_ID`).
3.  **Local project config:** `.gmconfig.yaml` in the current directory.
4.  **User global config:** `~/.config/gist_memory/config.yaml`.
5.  **Hardcoded defaults.**

You can manage the user global configuration using the `config` commands:
-   `gist-memory config show`: Displays the current effective configuration and where each value originates.
-   `gist-memory config set <KEY> <VALUE>`: Sets a key in the user global config file.

For example, to set your default memory path globally:
```bash
gist-memory config set gist_memory_path ~/my_gist_memories
# Subsequent commands will use this path unless overridden by --memory-path or an environment variable.
```
Or, set it using an environment variable for the current session:
```bash
export GIST_MEMORY_PATH=~/my_gist_memories
```

## Quick Start / Core Workflow

The `gist-memory` Command-Line Interface (CLI) is your primary tool for managing memory agents, ingesting data, querying, and summarizing.

**1. Initialize an Agent:**
First, create a new memory agent. This directory will store the agent's data.
```bash
gist-memory agent init ./my_agent --model-name sentence-transformers/all-MiniLM-L6-v2
```
This creates an agent at `./my_agent`. The specified embedding model will be downloaded if not already present.

**2. Configure Memory Path (Optional but Recommended for Convenience):**
To avoid specifying `--memory-path ./my_agent` for every command that interacts with this agent, you can set it globally for your user or for the current terminal session.

*   **Set globally (user config):**
    ```bash
    gist-memory config set gist_memory_path ./my_agent
    ```
    Now, `gist-memory` commands like `ingest` and `query` will default to using `./my_agent`.
*   **Set for current session (environment variable):**
    ```bash
    export GIST_MEMORY_PATH=$(pwd)/my_agent
    ```

**3. Ingest Data:**
Add information to the agent's memory.
```bash
# If gist_memory_path is set (globally or via env var):
gist-memory ingest path/to/your_document.txt
gist-memory ingest path/to/your_data_directory/

# Or, specify the memory path directly for a specific command:
gist-memory --memory-path ./my_agent ingest path/to/your_document.txt
```

**4. Query the Agent:**
Ask questions based on the ingested information. The agent uses its configured default model and strategy unless overridden.
```bash
# If gist_memory_path is set:
gist-memory query "What was mentioned about project X?"

# Or, specify the memory path directly:
gist-memory --memory-path ./my_agent query "What was mentioned about project X?"

# You can also override the default model or strategy for a specific query:
gist-memory query "Summarize recent findings on AI ethics" --model-id openai/gpt-4-turbo --strategy-id prototype
```

**5. Summarize Text (Standalone Utility):**
Compress text using a specific strategy without necessarily interacting with an agent's stored memory. This is useful for quick text compression tasks.
```bash
gist-memory summarize "This is a very long piece of text that needs to be shorter." --strategy first_last --budget 50
gist-memory summarize path/to/another_document.txt -s prototype -b 200 -o compressed_summary.txt
```

**Developer Tools & Evaluation:**
Gist Memory also includes tools for developers and researchers, such as evaluating compression strategies or testing LLM prompts. These are typically found under the `dev` subcommand group.

For example, to evaluate compression quality:
```bash
gist-memory dev evaluate-compression original.txt compressed_version.txt --metric compression_ratio
```
To list available strategies (including plugins):
```bash
gist-memory dev list-strategies
```

### Running Tests
To ensure Gist Memory is functioning correctly, especially after making changes or setting up the environment:
Install development dependencies (if not already done during setup):
```bash
pip install -r requirements.txt  # Ensure test dependencies are included
pytest
```

## Onboarding Demo

The `examples/onboarding_demo.py` script provides a practical demonstration of the Gist Memory experimentation workflow. It illustrates how to:

1.  Load a sample dataset.
2.  Apply different example `CompressionStrategy` implementations to this data.
3.  Feed the compressed output from these strategies to a simulated LLM.
4.  Showcase results from a simple `ValidationMetric` to compare the strategies.

To run the demo (ensure `gist-memory` is installed or `PYTHONPATH` is correctly set):
```bash
python examples/onboarding_demo.py
```

## Documentation

For more detailed information on Gist Memory's architecture, development guides, and advanced topics, please refer to the `docs/` directory.

-   **Main Documentation Portal:** `docs/README.md` (or `docs/index.md`) serves as a Table of Contents and entry point for deeper documentation.
-   **Architecture Deep Dive:** Understand the overall system design in `docs/ARCHITECTURE.md`.
-   **Developing Compression Strategies:** Learn how to create your own strategies in `docs/DEVELOPING_COMPRESSION_STRATEGIES.md`.
-   **Developing Validation Metrics:** Find guidance on building custom metrics in `docs/DEVELOPING_VALIDATION_METRICS.md`.
-   **Running Experiments:** Details on the experimentation framework can be found in `docs/RUNNING_EXPERIMENTS.md`.
-   **Sample Results:** Benchmarks produced with the framework are summarized in `RESULTS.md`.
-   **Plugins:** Learn how to install or develop strategy plugins in `docs/SHARING_STRATEGIES.md`.
-   **Conceptual Guides:** Explore the ideas behind memory strategies in `AGENTS.md`.

## Designing Compression Strategies

Gist Memory is designed to support a wide variety of `CompressionStrategy` implementations. For detailed guidance on creating your own, including best practices for splitting documents into meaningful chunks (e.g., belief-sized ideas) and techniques for updating memory (like centroid updates), please see:

-   `docs/COMPRESSION_STRATEGIES.md`
-   `docs/DEVELOPING_COMPRESSION_STRATEGIES.md`

The `AgenticChunker` is an example of an advanced chunking mechanism. You can enable it during agent initialization (e.g., `gist-memory agent init ./my_memory --chunker agentic`) or programmatically within your custom strategy (e.g., `agent.chunker = AgenticChunker()`).

## Query Tips

Effective querying is crucial when working with compressed memory. For tips on how to shape your search queries to best retrieve information, especially when your notes follow a structured format (like WHO/WHAT/WHEN), refer to:

-   `docs/QUERY_TIPS.md`

This document explains techniques such as templating your questions to align with the structure of your memory store, thereby biasing retrieval towards more relevant results.

## Architecture and Storage

Gist Memory features a modular architecture that allows for flexible extension and adaptation. The core interfaces for `CompressionStrategy` and `ValidationMetric` are designed to be pluggable, enabling diverse implementations.

Persistent storage, such as the default JSON/NPY store, is primarily relevant for `CompressionStrategy` implementations that maintain a stateful long-term memory (e.g., strategies based on prototypes or evolving summaries). Many strategies, however, can be stateless, processing input text without relying on persistent memory.

For a detailed architectural overview and discussion of storage options, please see:
-   `docs/ARCHITECTURE.md`

This document provides a high-level view of the components and their interactions.
