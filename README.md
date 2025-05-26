# Gist Memory Agent

Prototype implementation of the Gist Memory Agent using a coarse prototype memory system.

## Features

- CLI interface with `ingest` and `query` commands.
- Uses ChromaDB for persistent storage of prototypes and memories.
- Simple identity memory creation engine (pluggable).

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Ingest a memory:

```bash
python -m gist_memory ingest "Some text to remember"
```

Query memories:

```bash
python -m gist_memory query "search text" --top 5
```

Data is stored in `gist_memory_db` in the current working directory.
