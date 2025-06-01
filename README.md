# Gist Memory: An Experimentation Platform for LLM Memory Compression

**Gist Memory is a Python-based platform designed for rapidly prototyping, testing, and validating diverse strategies for compressing textual information ("memory") to maximize its utility within Large Language Model (LLM) token budgets.**

It provides a framework to implement and compare different approaches to making large text corpora or conversational history manageable and effective for LLMs.

## Core Features

* **Experimentation Framework:** Systematically test and validate memory compression strategies against defined datasets.
* **Pluggable `CompressionStrategy` Interface:** Easily implement and integrate diverse memory processing techniques. Examples include:
    * Active memory management (inspired by human working memory).
    * Gist-based prototype systems for long-term knowledge consolidation.
    * Extractive summarization, and more.
* **Pluggable `ValidationMetric` Interface:** Define and apply custom metrics to evaluate the effectiveness of compressed memory in LLM interactions (e.g., information recall, F1 score, coherence).
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

Alternatively, install the package from source:
```bash
pip install .
```
Run `gist-memory --help` to see available commands.

## Core Workflow
The platform facilitates the following general workflow:

1. **Ingest Text:** Provide a large text corpus or conversational data.
2. **Apply CompressionStrategy:** Choose and configure a strategy to process and compress the input text according to a specified token budget.
3. **Prepare LLM Prompt:** Combine the compressed memory with a user query/task.
4. **Interact with LLM:** Send the assembled prompt to an LLM.
5. **Validate Results:** Apply chosen ValidationMetric(s) to the LLM's response and the compressed context to evaluate effectiveness.

## Usage

### Running Experiments
The primary way to use Gist Memory is through its experimentation framework.

```bash
# Example: Run an experiment defined in a configuration file
gist-memory experiment --config path/to/experiment_config.yaml
```
(The CLI for experiment will need to be updated to accept CompressionStrategy, ValidationMetric, and their parameters).
Refer to [docs/RUNNING_EXPERIMENTS.md](docs/RUNNING_EXPERIMENTS.md) for details on setting up and interpreting experiments.

### Developing New Strategies & Metrics
Gist Memory is designed to be extensible:

* **To create a new CompressionStrategy:** Implement the CompressionStrategy interface (see `gist_memory/compression/strategies_abc.py` - to be created) and register it.
* **To create a new ValidationMetric:** Implement the ValidationMetric interface (see `gist_memory/validation/metrics_abc.py` - to be created) and register it.

Detailed guides will be available in:

* [docs/DEVELOPING_COMPRESSION_STRATEGIES.md](docs/DEVELOPING_COMPRESSION_STRATEGIES.md)
* [docs/DEVELOPING_VALIDATION_METRICS.md](docs/DEVELOPING_VALIDATION_METRICS.md)

### Basic CLI Interactions (for testing individual components)

Initialize a "Brain" (for LTM-based strategies): Some strategies (like the prototype-based one) might require initializing a persistent store.
```bash
gist-memory init my_ltm_store # Optional, if your strategy needs it
```

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

For deeper insights into the concepts behind some of the example strategies (like Active Memory and Prototypes), refer to the Conceptual Guide (`AGENTS.md`).

## Architecture and Storage

The Gist Memory platform is designed for modularity. Core interfaces for compression and validation allow for diverse implementations.

Persistent storage (like the JSON/NPY store) is primarily relevant for CompressionStrategy implementations that maintain a stateful long-term memory component (e.g., the prototype-based strategy). Other strategies might be stateless.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for a high-level overview.

