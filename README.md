# Gist Memory Agent

Prototype implementation of the Gist Memory Agent using a coarse prototype memory system.

## Features

- CLI interface with `init`, `add`, `query`, `list-beliefs`, `stats`,
  `validate`, `clear` and `download-model` commands.
- Lightweight JSON/NPY backend for prototypes and memories (default).
- Optional Chroma vector store for scale via ``pip install \"gist-memory[chroma]\"``.
- Pluggable memory creation engines (identity, extractive, chunk, LLM summary, or agentic splitting).
- Pluggable embedding backends: random (default), OpenAI, or local sentence-transformer.
- Chunks are rendered using a canonical ``WHO/WHAT/WHEN/WHERE/WHY`` template before embedding.
- Launches a simple Textual TUI when running `gist-memory` with no arguments.
- Python API provides helpers to decode and summarise prototypes.
- Chat with a brain using a local LLM via the `talk` command.
- Enable debug logging with `--log-file` or the `/log` TUI command.
- Conflicts are heuristically flagged and written to `conflicts.jsonl` for
  HITL review.

## Setup

This project requires **Python 3.11+**.  Install the dependencies and download
the default local embedding model (only needed the first time):

```bash
pip install -r requirements.txt
# download the spaCy model for sentence segmentation
python -m spacy download en_core_web_sm
# fetch the "all-MiniLM-L6-v2" model so the local embedder works offline
gist-memory download-model --model-name all-MiniLM-L6-v2
# fetch the default chat model for talk mode
gist-memory download-chat-model --model-name distilgpt2
```

For a quick offline setup you can also run:

```bash
bash .codex/setup.sh
```

Once installed, running `gist-memory` with no arguments will start the Textual TUI.

Alternatively install the package from source:

```bash
pip install .
```

## Usage

### Interactive TUI

Launch ``gist-run`` to explore a brain interactively:

```bash
gist-run
```

All commands operate on the ``brain`` directory by default. Pass
``--agent-name`` (or a directory argument for ``gist-memory init``) to use a
different location.

### Command line

Initialise a new brain then ingest a memory:

```bash
gist-memory init brain
gist-memory add --text "Some text to remember"
```

When using the OpenAI embedder, set the ``OPENAI_API_KEY`` environment
variable so the library can authenticate.

You can also pass a path to a text file or a directory containing ``*.txt``
files:

```bash
gist-memory add --file notes.txt
gist-memory add --file docs/
```

Query memories:

```bash
gist-memory query --query-text "search text" --k-memories 5
```

Chat with the entire brain using a local model:

```bash
gist-memory talk --message "What's in this brain?"
```
Ensure the chat model is pre-downloaded using `gist-memory download-chat-model`.

List belief prototypes and show store stats:

```bash
gist-memory list-beliefs
gist-memory stats
gist-memory validate
# permanently remove all data
gist-memory clear --yes
```

Additional functions such as decoding or summarising a prototype are
available via the Python API.

The local embedder loads the model from the Hugging Face cache only and will not
attempt any network downloads. Ensure the embedding and chat models are
pre-cached using the commands in the setup section or via `.codex/setup.sh`.

Data is stored in the `brain` directory by default in the current working directory.

## Scaling with Chroma

If you exceed about **10k beliefs** or plan to run multi-process agents, switch to
the Chroma backend:

```yaml
vector_store: chroma
chroma_path: ./belief_db
```

Install the optional dependency first:

```bash
pip install "gist-memory[chroma]"
```

A quick benchmark shows brute-force JSON lookup taking ~20 ms for 1k beliefs
versus <5 ms with Chroma. Larger datasets benefit even more.

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
gist-memory download-model --model-name all-MiniLM-L6-v2

# run the example
python examples/onboarding_demo.py
```

The script loads all `*.txt` files from `examples/moon_landing`, stores them in a
local database and displays the prototype assignments along with a final memory
and prototype count.

## Wizard Demo

The Textual wizard provides a hands-on tour of the agent. Install the
requirements, download the embedding model and launch ``gist-run``:

```bash
pip install -r requirements.txt
gist-memory download-model --model-name all-MiniLM-L6-v2

# start the wizard
gist-run
```

From the welcome screen press ``L`` to load the sample Apollo 11 transcripts
from ``examples/moon_landing``. The wizard now uses a small console where you
interact via slash commands. Type ``/ingest <text>`` to add a memory, ``/query
<text>`` to search, ``/stats`` to see store statistics, ``/install-models`` to
download the local models or ``/help`` for a list of commands.

## Segmentation Playbook

See [docs/SEGMENTATION_PLAYBOOK.md](docs/SEGMENTATION_PLAYBOOK.md) for a detailed workflow on splitting long documents into belief-sized ideas before ingestion. You can enable this behaviour in the CLI via `--memory-creator agentic`.

## Querying Playbook

See [docs/QUERYING_PLAYBOOK.md](docs/QUERYING_PLAYBOOK.md) for tips on shaping search queries. It explains how to bias retrieval by embedding a templated version of a question when your notes follow a structured `WHO/WHAT/WHEN` format.
