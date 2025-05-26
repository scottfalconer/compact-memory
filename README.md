# Gist Memory Agent

Prototype implementation of the Gist Memory Agent using a coarse prototype memory system.

## Features

- CLI interface with `ingest` and `query` commands.
- Uses ChromaDB for persistent storage of prototypes and memories.
- Pluggable memory creation engines (identity or simple extractive summary).
- Pluggable embedding backends: random (default), OpenAI, or local sentence-transformer.

## Usage

Install dependencies (the provided `.codex/setup.sh` script will also install
them and download the default local embedding model for offline use):

```bash
pip install -r requirements.txt
python - <<'EOF'
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2")
EOF
```

Ingest a memory:

```bash
python -m gist_memory ingest "Some text to remember" \
    --embedder openai --memory-creator extractive --threshold 0.3
```

Query memories:

```bash
python -m gist_memory query "search text" --top 5 \
    --embedder local --model-name all-MiniLM-L6-v2 --threshold 0.3
```

The local embedder loads the model from the Hugging Face cache only and will not
attempt any network downloads. Ensure the model is pre-cached using the commands
in the installation section or via `.codex/setup.sh`.

Data is stored in `gist_memory_db` in the current working directory.

## Running Tests

Install development dependencies and run `pytest`:

```bash
pip install -r requirements.txt
pytest
```
