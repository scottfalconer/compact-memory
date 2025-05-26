# Gist Memory Agent

Prototype implementation of the Gist Memory Agent using a coarse prototype memory system.

## Features

- CLI interface with `ingest` and `query` commands.
- Uses ChromaDB for persistent storage of prototypes and memories.
- Simple identity memory creation engine (pluggable).
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
python -m gist_memory ingest "Some text to remember" --embedder openai
```

Query memories:

```bash
python -m gist_memory query "search text" --top 5 --embedder local --model-name all-MiniLM-L6-v2
```

Data is stored in `gist_memory_db` in the current working directory.

## Running Tests

Install development dependencies and run `pytest`:

```bash
pip install -r requirements.txt
pytest
```
