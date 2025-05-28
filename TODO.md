# TODO

This file tracks outstanding work based on `AGENTS.md` and the latest v3 implementation plan.

## 1 · Data-Model & Storage Layer
- Implement `BeliefPrototype` and `RawMemory` Pydantic models.
- Validate `meta.yaml` for embedding model name and dimension.
- Create a `VectorStore` ABC exposing add/update/query/save/load.
- Implement `JsonNpyVectorStore` with row-order mapping and `gist migrate --to json`.

## 2 · Embedding & Chunking
- Provide `embed_text()` wrapper using the local `all-MiniLM-L6-v2` model.
- Add a `Chunker` interface with NLTK-based sentence splitting.
- Detect duplicate memories using SHA-256 hashes.

## 3 · Ingestion & Prototype Logic
- Implement `Agent.add_memory()` that chunkes, embeds and searches.
- Use a fixed threshold τ = 0.8 for now and document tuning TODOs.
- Spawn new prototypes only when all similarities are below τ.

## 4 · Retrieval & CLI
- Implement `agent.query()` returning the nearest prototype and top-N memories.
- Add CLI commands: `gist init`, `add`, `query`, `list-beliefs`, `stats`.
- Show counts and file sizes in `gist stats`.

## 5 · Testing & CI
- Add a mock encoder for deterministic vectors in tests.
- Provide fixtures with small sentences expecting two prototypes.
- Setup GitHub Actions for linting and running unit tests.

## 6 · Memory Decay / Archival
- Implement `agent.maintain_memory()` with exponential decay.
- Archive memories with score below ε and mark empty prototypes dormant.

## 7 · Multi-Agent & Trust
- Load a static trust table from agent YAML; default 1.0 self, 0.5 unknown.
- Implement `teach`/`learn` API scaling α by trust.
- Add CLI support for `gist agent teach/learn`.

## 8 · Conflict Flagging (V1)
- Heuristically flag contradictions using simple negation checks.
- Record conflicts in a JSON Lines log for later review.

## 9 · Documentation & OSS Polish
- Write `docs/architecture.md` describing the storage format and workflow.
- Document the JSON store versioning rules in `docs/storage_format.md`.
- Update README with a quick-start example and security note.
- List third‑party license notices.

## 10 · Future-Facing Stubs
- Stub out `ChromaVectorStore` with `NotImplemented` methods.
- Leave TODOs for advanced health checks and split/merge logic.
