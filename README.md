# Gist Memory Agent

Prototype implementation of the Gist Memory Agent using a coarse prototype memory system.

## Features

- CLI interface with `ingest`, `query`, `decode`, `summarize`, and `dump` commands.
- Uses ChromaDB for persistent storage of prototypes and memories.
- Pluggable memory creation engines (identity, extractive, chunk, LLM summary, or agentic splitting).
- Pluggable embedding backends: random (default), OpenAI, or local sentence-transformer.
- Ingest operation displays a progress bar showing which prototypes are created or updated.
- Launches a simple Textual TUI when running `gist-memory` with no arguments.

## Setup

This project requires **Python 3.11+**.  Install the dependencies and download
the default local embedding model (only needed the first time):

```bash
pip install -r requirements.txt
# fetch the "all-MiniLM-L6-v2" model so the local embedder works offline
python -m gist_memory download-model --model-name all-MiniLM-L6-v2

# You can alternatively run `.codex/setup.sh` to perform these steps.
```

Alternatively install the package from source:

```bash
pip install .
```

## Usage

### Interactive TUI

Run ``gist-memory`` with no arguments to start an in-terminal interface that
allows you to pick a ``.txt`` file from the current directory or enter text
manually for ingestion:

```bash
gist-memory
```

### Command line

Ingest a memory:

```bash
python -m gist_memory \
    --embedder openai --memory-creator extractive --threshold 0.3 \
    --min-threshold 0.05 --decay-exponent 0.5 \
    ingest "Some text to remember"
```

When using the OpenAI embedder, set the ``OPENAI_API_KEY`` environment
variable so the library can authenticate.

You can also pass a path to a text file or a directory containing ``*.txt``
files:

```bash
python -m gist_memory ingest notes.txt
python -m gist_memory ingest docs/
```

Query memories:

```bash
python -m gist_memory \
    --embedder local --model-name all-MiniLM-L6-v2 --threshold 0.3 \
    query "search text" --top 5
```

Decode a prototype to see example memories:

```bash
python -m gist_memory decode <prototype_id> --top 2
```

Summarize a prototype:

```bash
python -m gist_memory summarize <prototype_id> --max-words 20
```

Dump all memories (optionally filter by prototype):

```bash
python -m gist_memory dump --prototype-id <prototype_id>
```

The local embedder loads the model from the Hugging Face cache only and will not
attempt any network downloads. Ensure the model is pre-cached using the commands
in the setup section or via `.codex/setup.sh`.

Data is stored in `gist_memory_db` in the current working directory.

## Running Tests

Install development dependencies and run `pytest`:

```bash
pip install -r requirements.txt
pytest
```

## Onboarding Demo

A small example is provided under `examples/` so new users can quickly see the
agent in action.  The demo ingests short excerpts from the Apollo&nbsp;11 moon
landing transcripts and prints which prototypes were created or updated.

Run the demo after installing dependencies and pre-downloading the local
embedding model:

```bash
pip install -r requirements.txt
# download the local model once
python -m gist_memory download-model --model-name all-MiniLM-L6-v2

# run the example
python examples/onboarding_demo.py
```

The script loads all `*.txt` files from `examples/moon_landing`, stores them in a
local database and displays the prototype assignments along with a final memory
and prototype count.

## Segmentation Playbook

See [docs/SEGMENTATION_PLAYBOOK.md](docs/SEGMENTATION_PLAYBOOK.md) for a detailed workflow on splitting long documents into belief-sized ideas before ingestion. You can enable this behaviour in the CLI via `--memory-creator agentic`.
