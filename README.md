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

- Command-line interface for initialization, stats, validation, downloads, experiments, and strategy inspection.
- Lightweight JSON/NPY backend for prototypes and memories with optional Chroma vector store for scale (`pip install "gist-memory[chroma]"`).
- Pluggable memory creation engines (identity, extractive, chunk, LLM summary, or agentic splitting).
- Pluggable embedding backends: random (default), OpenAI, or local sentence transformers.
- Chunks rendered using a canonical **WHO/WHAT/WHEN/WHERE/WHY** template before embedding.
- Runs smoothly in Colab; a notebook-based GUI is planned.
- Python API for decoding and summarizing prototypes.
- Local chat interface via the `talk` command.
- Debug logging with `--log-file` and conflict heuristics written to `conflicts.jsonl` for HITL review.

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
    gist-memory download-model --model-name all-MiniLM-L6-v2
    # Fetch a default chat model for the 'talk' command and LLM-based validation
    gist-memory download-chat-model --model-name tiny-gpt2
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

## Core Workflow

The `gist-memory` Command-Line Interface (CLI) is your primary tool for experimenting with different compression strategies and evaluating their effectiveness. Here are some examples of common operations:

**1. Basic Compression:**
Compress a piece of text using a chosen strategy and token budget.
```bash
gist-memory compress "Your large text here... or path/to/file.txt" --strategy <strategy_name> --budget 500
```
This command will output the compressed version of your text. Replace `<strategy_name>` with an available strategy (e.g., `extractive`, `summary_llm`).

**2. Interactive Chat with Compressed Context:**
Engage in a conversation where the LLM's context is managed by a compression strategy.
```bash
gist-memory talk --strategy <strategy_name> --message "What can you tell me based on the compressed context?"
```
The LLM will use the compressed memory generated by the specified strategy to answer your questions.

**3. Direct Evaluation of Compression:**
Assess how much a text was compressed.
```bash
gist-memory evaluate-compression original.txt compressed.txt --metric compression_ratio --json
```
This compares the original text with its compressed version using a specific metric like `compression_ratio`.

**4. Pipeline Evaluation (Compress -> Prompt LLM -> Evaluate Response):**
A more complex workflow that simulates a full cycle: compress information, use it to prompt an LLM, and then evaluate the LLM's response.
```bash
gist-memory compress file.txt --strategy none --budget 50 --output-trace trace.json \
  | gist-memory llm-prompt --context - --query "Summarize the provided text." --model tiny-gpt2 \
  | gist-memory evaluate-llm-response - "An accurate summary would be..." --metric exact_match --json
```
This example:
- Compresses `file.txt` using a (hypothetical) `none` strategy with a budget of 50 tokens, saving a trace.
 - Pipes the compressed output (`-`) to `llm-prompt` to ask a tiny-gpt2 model to summarize it.
- Pipes the LLM's response (`-`) to `evaluate-llm-response` to check if it matches an expected summary using the `exact_match` metric.

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

The `AgenticChunker` is an example of an advanced chunking mechanism. You can enable it during memory initialization (e.g., `gist-memory init brain --chunker agentic`) or programmatically within your custom strategy (e.g., `agent.chunker = AgenticChunker()`).

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
