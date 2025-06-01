# Gist Memory Agent

Prototype implementation of the Gist Memory Agent using a coarse prototype memory system.

## Features

- CLI interface with `init`, `stats`, `validate`, `clear`, `download-model`,
  `download-chat-model`, `experiment` and `strategy inspect` commands.
- Lightweight JSON/NPY backend for prototypes and memories (default).
- Optional Chroma vector store for scale via ``pip install \"gist-memory[chroma]\"``.
- Pluggable memory creation engines (identity, extractive, chunk, LLM summary, or agentic splitting).
- Pluggable embedding backends: random (default), OpenAI, or local sentence-transformer.
- Chunks are rendered using a canonical ``WHO/WHAT/WHEN/WHERE/WHY`` template before embedding.
- The CLI runs smoothly in Colab.
- A Colab notebook will provide an interactive GUI in the future.
- Python API provides helpers to decode and summarise prototypes.
- Chat with a brain using a local LLM via the `talk` command.
- Enable debug logging with `--log-file`.
- Conflicts are heuristically flagged and written to `conflicts.jsonl` for
  HITL review.


**Gist Memory is a Python-based platform designed for rapidly prototyping, testing, and validating diverse strategies for compressing textual information ("memory") to maximize its utility within Large Language Model (LLM) token budgets.**

It provides a framework to implement and compare different approaches to making large text corpora or conversational history manageable and effective for LLMs.

## Core Features

* **Experimentation Framework:** Systematically test and validate memory compression strategies against defined datasets.
* **Pluggable `CompressionStrategy` Interface:** Easily implement and integrate diverse memory processing techniques. Examples include:
    * Active memory management (inspired by human working memory).
    * Gist-based prototype systems for long-term knowledge consolidation.
    * Extractive summarization, and more.
* **Pluggable `ValidationMetric` Interface:** Define and apply custom metrics to evaluate the effectiveness of compressed memory in LLM interactions (e.g., information recall, ROUGE, BLEU). Metrics can leverage the Hugging Face `evaluate` library.
* **Command-Line Interface (CLI):** Manage experiments, test strategies, and interact with the system.
* **Local LLM Interaction:** Test compressed memory with local LLMs for end-to-end validation.
* **Flexible Storage Backend:** Includes a lightweight JSON/NPY backend for components of strategies that require persistence (like LTM prototypes). (ChromaDB support can be mentioned if it remains relevant for specific strategies).

## Why Gist Memory?

* **Researchers:** Benchmark and compare novel memory compression algorithms for LLMs.
* **Developers:** Find the most effective way to fit large amounts of contextual data into limited LLM prompt windows for your application.
* **Community:** Share and discover new techniques for efficient LLM memory management.

## Setup

This project requires **Python 3.11+**.

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    # Download the spaCy model for sentence segmentation (used by some strategies/chunkers)
    python -m spacy download en_core_web_sm
    ```

2.  **Download Models for Testing (Optional but Recommended):**
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

3.  **Set API Keys for Cloud Providers:**
    If you plan to use OpenAI or Gemini models, export your credentials as
    environment variables before running the CLI:
    ```bash
    export OPENAI_API_KEY="sk-..."
    export GEMINI_API_KEY="..."
    ```

Alternatively, install the package from source:
```bash
pip install .
```
Run `gist-memory --help` to see available commands.

## Core Workflow
The platform facilitates the following general workflow:



Test a Compression Strategy (New/Revised Command):
```bash
gist-memory compress --strategy <strategy_name> --text "Your large text here..." --budget 500
```

Chat with Compressed Memory (Revised Command): The talk command can be used to test how an LLM responds when provided with context from a specific compression strategy.
```bash
gist-memory talk --strategy <strategy_name> --message "What can you tell me based on the compressed context?"
```

### Running Tests (for contributors)
Install development dependencies and run pytest:
```bash
pip install -r requirements.txt
pytest
```

### Onboarding Demo (Revised)
The `examples/onboarding_demo.py` script will be updated to showcase the experimentation workflow. For example, it might:

1. Load a sample dataset.
2. Apply two different example CompressionStrategy implementations.
3. Feed the compressed output to a dummy LLM.
4. Show results from a simple ValidationMetric.

Run the demo (after setup):
```bash
python examples/onboarding_demo.py
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

