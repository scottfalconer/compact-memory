# TODO

This file tracks outstanding work based on the earlier project vision and the latest v3 implementation plan.

## 2 · Embedding & Chunking

## 3 · Ingestion & Prototype Logic
- Use a fixed threshold τ = 0.8 for now and document tuning TODOs.
- Spawn new prototypes only when all similarities are below τ.

## 4 · Retrieval & CLI
- Implement `agent.query()` returning the nearest prototype and top-N memories.
- Add CLI commands: `gist init`, `add`, `query`, `list-beliefs`, `stats`.
- Show counts and file sizes in `gist stats`.

## 5 · Testing & CI
- Provide fixtures with small sentences expecting two prototypes.
- Setup GitHub Actions for linting and running unit tests.
- Add unit tests for `Agent.query` covering success and "no_match" cases.
- Extend CLI tests to cover `list-beliefs` and `download-model` commands.
- Add embedding pipeline tests for single-string and empty inputs, and for `register_embedding`.

## 6 · Memory Decay / Archival
- Implement `agent.maintain_memory()` with exponential decay.
- Archive memories with score below ε and mark empty prototypes dormant.

## 7 · Multi-Agent & Trust
- Load a static trust table from agent YAML; default 1.0 self, 0.5 unknown.
- Implement `teach`/`learn` API scaling α by trust.
- Add CLI support for `gist agent teach/learn`.

## 8 · Conflict Flagging (V1) - NEEDS CLARIFICATION/RESEARCH
- NEEDS CLARIFICATION: Further research is needed to define heuristic methods for contradiction detection and what specifically constitutes a 'contradiction' in this context.

## 9 · Documentation & OSS Polish
- Write `ARCHITECTURE.md` describing the storage format and workflow.
- Document the JSON store versioning rules in `STORAGE_FORMAT.md`.

## 10 · Future-Facing Stubs
- Leave TODOs for advanced health checks and split/merge logic.
