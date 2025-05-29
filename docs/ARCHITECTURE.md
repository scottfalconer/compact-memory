# Gist Memory Architecture

## Overview

`gist-memory` implements a coarse prototype memory system for storing and
retrieving textual information. Incoming text is converted into vector
embeddings and "snap-assigned" to belief prototypes which act as stable
centroids. The codebase is organised into a library providing Python APIs,
a CLI (`gist-memory`) and a lightweight Textual TUI (`gist-run`).

The design follows the hypotheses documented in `AGENTS.md`:
prototypes reduce storage and search cost while providing more robust
gist-based reasoning. The implementation emphasises pluggability so
alternative memory creation, embedding and storage mechanisms can be
experimented with.

## Package layout

```
├── gist_memory/           # main library
├── gist_tui/              # thin wrapper running the TUI
├── examples/              # onboarding demo
├── tests/                 # unit tests using mock embeddings
└── docs/                  # documentation
```

### Core modules

- **`agent.py`** – orchestrates ingestion and querying using a
  `JsonNpyVectorStore`. It handles deduplication, chunking, embedding and
  prototype updates. The default similarity threshold is 0.8.
- **`json_npy_store.py`** – minimal persistence layer storing prototype
  vectors in NPY files and metadata in JSON. Provides nearest-neighbour
  search and exposes methods used by the agent to add/update prototypes
  and memories.
- **`embedding_pipeline.py`** – loads a SentenceTransformer model and
  exposes `embed_text`. A deterministic `MockEncoder` is available for
  tests. Embeddings are cached and normalised.
- **`chunker.py`** – implements sentence-window based chunking with token
  overlap and a fixed-size fallback. The registry allows different
  chunkers to be plugged in via config.
- **`memory_creation.py`** – small utilities to create "memory" texts
  from raw documents. Includes identity, extractive, fixed chunk and
  LLM-driven variants so experiments can measure which produces better
  prototypes.
- **`cli.py`** – Typer-based command line app supporting `init`, `add`,
  `query`, `list-beliefs`, `stats` and `download-model`. Persistence is
  locked during writes to avoid corruption.
- **`tui.py`** – interactive Textual interface for exploring a brain.
  Provides screens for ingesting text, listing beliefs, querying and
  viewing statistics.
- **`embedding_pipeline.py`**, **`embedder.py`** and **`store.py`** –
  additional abstractions for embedding backends and alternative vector
  stores (e.g. ChromaDB). These allow switching to different storage or
  model providers without rewriting the agent logic.

### Data models

`models.py` defines two `pydantic` models:

- `BeliefPrototype` – metadata about a prototype without the vector.
  Tracks strength, confidence, timestamps and the IDs of constituent
  memories.
- `RawMemory` – a chunk of source text with optional embedding and the
  prototype it belongs to.

They are stored in JSON and referenced by ID in the NPY vector arrays.

## Ingestion flow

1. **Chunking** – text is split into sentence windows using
   `SentenceWindowChunker` (or another registered chunker).
2. **Embedding** – each chunk is converted into a normalised vector via
   `embed_text`.
3. **Prototype search** – the store finds the nearest prototype by
   cosine similarity. If no similarity exceeds the threshold the agent
   spawns a new prototype.
4. **Updating** – existing prototypes are updated using an exponential
   moving average. New memories are appended to the JSON lines file and
   evidence is logged in `evidence.jsonl`.

The agent keeps an LRU cache of recent SHA‑256 hashes to skip duplicates
quickly.

## Querying

1. The query text is embedded and the top `k` prototypes are retrieved.
2. Constituent memories from those prototypes are scored by similarity to
   the query embedding.
3. The results include the ranked prototypes and the top memories.

## CLI/TUI

- `gist-memory` invokes the CLI. Without arguments it falls back to the
  TUI via `__main__.py`.
- `gist-run` is an entry point that launches the wizard-like TUI defined
  in `tui.py`.

Both interfaces operate on the same underlying store structure.

## Testing

The test suite uses the deterministic `MockEncoder` to avoid heavy model
loads. `pytest` exercises the agent logic, CLI commands, chunkers, the
embedding pipeline and both vector stores. Continuous integration ensures
that persistence round‑trips and query ranking work as expected.

## Rationale

The current implementation aims to validate the coarse prototype memory
hypothesis with a lightweight, easily inspectable codebase. JSON/NPY
storage avoids external dependencies while still supporting millions of
vectors. Abstractions for embedding models and vector stores allow the
system to evolve without rewriting the agent logic. Unit tests with a
mock encoder keep the feedback loop fast and deterministic.

