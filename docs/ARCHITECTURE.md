# Gist Memory Architecture

This document provides a detailed technical explanation of the Gist Memory platform's architecture, intended for developers contributing to the core or those seeking a deep understanding of its internal workings.

## Overview

`gist-memory` implements a coarse prototype memory system for storing and
retrieving textual information. Incoming text is converted into vector
embeddings and "snap-assigned" to belief prototypes which act as stable
centroids. The codebase is organised into a library providing Python APIs and
a CLI (`gist-memory`).

The design follows the hypotheses documented in `AGENTS.md`:
prototypes reduce storage and search cost while providing more robust
gist-based reasoning. The implementation emphasises pluggability so
alternative memory creation, embedding and storage mechanisms can be
experimented with.

## Package layout

```
├── gist_memory/           # main library
├── examples/              # onboarding demo
├── tests/                 # unit tests using mock embeddings
└── docs/                  # documentation
```

### Core modules
<!-- SUGGESTION: A diagram illustrating the interaction between core modules (agent.py, json_npy_store.py, embedding_pipeline.py, chunker.py, etc.) could be helpful here. -->

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
- **`cli.py`** – Typer-based command line app supporting `init`,
  `stats`, `validate`, `clear`, `download-model`, `download-chat-model`,
  `experiment` and `strategy inspect`. Persistence is
  locked during writes to avoid corruption.
- Additional vector store implementations can be developed by
  extending the interfaces used by `JsonNpyVectorStore`. This keeps
  the agent decoupled from any particular storage backend or embedding
  provider.

### Data models
<!-- SUGGESTION: A simple diagram showing the relationship between BeliefPrototype and RawMemory data models would be useful. -->

`models.py` defines two `pydantic` models:

- `BeliefPrototype` – metadata about a prototype without the vector.
  Tracks strength, confidence, timestamps and the IDs of constituent
  memories.
- `RawMemory` – a chunk of source text with optional embedding and the
  prototype it belongs to.

They are stored in JSON and referenced by ID in the NPY vector arrays.

## Ingestion flow
<!-- SUGGESTION: A diagram illustrating the ingestion flow (Chunking -> Embedding -> Prototype Search -> Updating) would be beneficial here. -->

1. **Chunking** – text is split into sentence windows using
   `SentenceWindowChunker` or into belief-sized ideas with
   `AgenticChunker` (both are registered chunkers).
2. **Embedding** – each chunk is converted into a normalised vector via
   `embed_text`.
3. **Prototype search** – the store finds the nearest prototype by
   cosine similarity. If no similarity exceeds the threshold the agent
   spawns a new prototype.
4. **Updating** – existing prototypes are updated using an exponential
   moving average. New memories are appended to the JSON lines file and
   evidence is logged in `evidence.jsonl`. Potential contradictions
   detected via a simple negation check are appended to
   `conflicts.jsonl` for human review.

The agent keeps an LRU cache of recent SHA‑256 hashes to skip duplicates
quickly.

## Querying
<!-- SUGGESTION: A diagram illustrating the querying process (Query Embedding -> Prototype Retrieval -> Memory Scoring) would be beneficial here. -->

1. The query text is embedded and the top `k` prototypes are retrieved.
2. Constituent memories from those prototypes are scored by similarity to
   the query embedding.
3. The results include the ranked prototypes and the top memories.

## CLI

`gist-memory` is implemented using Typer and exposes subcommands for
initialising a brain, inspecting stored prototypes and running
experiments. The CLI is the primary interface, and a Colab notebook will
provide a graphical option in the future.

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

