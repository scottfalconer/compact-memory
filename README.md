# Gist Memory

Gist Memory: An Experimentation Platform for Advanced LLM Memory Strategies

Large Language Models (LLMs) are transforming how we interact with information, yet their ability to maintain long-term, coherent, and efficient memory remains a significant challenge. Standard approaches like Retrieval Augmented Generation (RAG) and expanding context windows offer partial solutions, but often fall short for complex tasks requiring dynamic learning, nuanced understanding, and resource-conscious operation.

Gist Memory is an open-source platform dedicated to pioneering the next generation of LLM memory.

It provides a robust framework for researchers and developers to design, rigorously test, and validate sophisticated CompressionStrategy implementations. Instead of just retrieving static data, Gist Memory facilitates the exploration of advanced techniques, including:

- **Evolving Gist-Based Understanding:** Strategies where memory consolidates, adapts, and forms conceptual "gists" from information over time.
- **Resource-Efficient Context Optimization:** Methods to maximize information density within token budgets, critical for managing API costs, latency, and enabling smaller or local LLMs.
- **Learned Compression and Summarization:** Techniques that can be trained or adapt to optimally compress information for specific tasks or data types.
- **Active Memory Management:** Systems that simulate dynamic working memory, managing recency, relevance, and trace strength for coherent, long-running interactions.

Gist Memory is built for those pushing the boundaries of LLM capabilities:

- Researchers seeking a standardized environment to benchmark novel memory architectures and share reproducible findings.
- Developers aiming to equip their LLM applications with more powerful, adaptive, and efficient memory than off-the-shelf solutions provide.

This platform offers pluggable interfaces for memory strategies and validation metrics, a comprehensive Command-Line Interface (CLI) for experiment management, and the tools necessary to systematically evaluate and iterate on the future of LLM memory.

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

* **Researchers:** Benchmark and compare novel memory compression algorithms for LLMs.
* **Developers:** Find the most effective way to fit large amounts of contextual data into limited LLM prompt windows for your application.
* **Community:** Share and discover new techniques for efficient LLM memory management.

* Go beyond standard Retrieval Augmented Generation (RAG) by creating an *evolving understanding*. Gist Memory allows strategies that consolidate, update, and form conceptual gists from information over time.
* Optimize LLM interactions in resource-constrained settings, reducing token counts and enabling smaller or local models.
* Facilitate research into *learned compression*, testing strategies that adapt and improve based on data.

## Setup
This project requires **Python 3.11+**.

0.  **Run `setup.sh` (optional):**
    To prepare an offline-friendly environment quickly, execute the provided
    script while you still have internet access:
    ```bash
    ./setup.sh
    ```

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    # Download the spaCy model for sentence segmentation (used by some strategies/chunkers)
    python -m spacy download en_core_web_sm
    ```

2.  **Install `gist-memory`** – required for the CLI and examples.
    When working in an offline environment use `--no-build-isolation` or simply
    add the repository to your `PYTHONPATH`:
    ```bash
    pip install -e . --no-build-isolation  # or
    export PYTHONPATH="$(pwd):$PYTHONPATH"
    ```

3.  **Download Models for Testing (Optional but Recommended):**
    These models are used by example strategies or for testing LLM interactions with compressed memory.
    ```bash
    # Fetch the "all-MiniLM-L6-v2" model for embedding (used by default LTM components)
    gist-memory download-model --model-name all-MiniLM-L6-v2
    # Fetch a default chat model for the 'talk' command and LLM-based validation
    gist-memory download-chat-model --model-name distilgpt2
    ```
    To run completely offline after downloads, set:
    ```
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    ```

4.  **Set API Keys for Cloud Providers (Optional):**
    If you plan to use OpenAI or Gemini models, export your credentials as environment variables before running the CLI:
    ```bash
    export OPENAI_API_KEY="sk-..."
    export GEMINI_API_KEY="..."
    ```
Run `gist-memory --help` to see available commands.

## Core Workflow
Use the CLI to experiment with different compression strategies:

```bash
gist-memory compress "Your large text here..." --strategy <strategy_name> --budget 500
gist-memory talk --strategy <strategy_name> --message "What can you tell me based on the compressed context?"
```

### Running Tests
Install development dependencies and run pytest:
```bash
pip install -r requirements.txt
pytest
```

### Onboarding Demo
The `examples/onboarding_demo.py` script demonstrates the experimentation workflow:

1. Load a sample dataset.
2. Apply example CompressionStrategy implementations.
3. Feed the compressed output to a dummy LLM.
4. Show results from a simple ValidationMetric.

Run the demo (after setup):
```bash
python examples/onboarding_demo.py  # ensure `gist-memory` is installed or PYTHONPATH is set
```
## Key Concepts


* **CompressionStrategy:** An algorithm that takes input text and a token budget, and produces a compressed representation of that text suitable for an LLM prompt.
* **ValidationMetric:** A method to evaluate the quality or utility of the compressed memory, often by assessing an LLM's performance on a task using that compressed memory.
* **Experimentation Framework:** The tools and processes within Gist Memory for systematically running tests with different strategies, datasets, and metrics.

## Designing Compression Strategies

See [docs/COMPRESSION_STRATEGIES.md](docs/COMPRESSION_STRATEGIES.md) for guidance on splitting documents into belief-sized ideas and updating centroids. The advanced `AgenticChunker` implements these techniques and can be enabled via `gist-memory init brain --chunker agentic` or programmatically with `agent.chunker = AgenticChunker()`.

## Query Tips

[docs/QUERY_TIPS.md](docs/QUERY_TIPS.md) explains how to shape search queries. When your notes follow a structured `WHO/WHAT/WHEN` layout you can embed a templated version of the question to bias retrieval.
For deeper insights into the concepts behind some of the example strategies (like Active Memory and Prototypes), refer to the Conceptual Guide (`AGENTS.md`).

## Architecture and Storage

The Gist Memory platform is designed for modularity. Core interfaces for compression and validation allow for diverse implementations.

Persistent storage (like the JSON/NPY store) is primarily relevant for CompressionStrategy implementations that maintain a stateful long-term memory component (e.g., the prototype-based strategy). Other strategies might be stateless.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for a high-level overview.

