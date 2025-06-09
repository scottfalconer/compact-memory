# TODO

This file tracks outstanding work based on the earlier project vision and the latest v3 implementation plan.

## 2 · Embedding & Chunking
- Explore advanced chunking strategies beyond fixed size/overlap, focusing on semantically meaningful units (e.g., sentence/paragraph boundaries, coherence metrics).

## 3 · Ingestion & Prototype Logic
- Use a fixed threshold τ = 0.8 for now and document tuning TODOs.
- Spawn new prototypes only when all similarities are below τ.
- Enhance memory traces with richer contextual metadata: e.g., timestamp, source, links to memories active during encoding, salience scores.
- Implement an "ingestion confidence score" based on parseability, clarity, or ambiguity of new data.

## 4 · Retrieval & CLI
- Implement `agent.query()` returning the nearest prototype and top-N memories.
- Add CLI commands: `gist init`, `add`, `query`, `list-beliefs`, `stats`.
- Show counts and file sizes in `gist stats`.
- Return a "confidence/relevance score" with query results, based on similarity, memory strength, and contextual match.
- Explore context-dependent retrieval: allow `query()` to use metadata (timestamps, sources, links, or even recent working memory context) as cues, potentially drawing inspiration from how cognitive systems use context in memory recall (e.g., aspects of `neocortex_transfer.py`'s retrieval).

## 5 · Testing & CI
- Provide fixtures with small sentences expecting two prototypes.
- Setup GitHub Actions for linting and running unit tests.
- Add unit tests for `Agent.query` covering success and "no_match" cases.
- Extend CLI tests to cover `list-beliefs` and `download-model` commands.
- Add embedding pipeline tests for single-string and empty inputs, and for `register_embedding`.

## 6 · Memory Decay / Archival
- Implement `agent.maintain_memory()` with exponential decay.
- Archive memories with score below ε and mark empty prototypes dormant.
- Implement reinforcement through retrieval: successful queries should strengthen memories (e.g., reset decay, update access counters/timestamps).

## 7 · Multi-Agent & Trust
- Load a static trust table from agent YAML; default 1.0 self, 0.5 unknown.
- Implement `teach`/`learn` API scaling α by trust.
- Add CLI support for `gist agent teach/learn`.

## 8 · Conflict Flagging (V1) - NEEDS CLARIFICATION/RESEARCH
- NEEDS CLARIFICATION: Further research is needed to define heuristic methods for contradiction detection and what specifically constitutes a 'contradiction' in this context.
- Research and implement "novelty/anomaly scoring" for new data relative to existing knowledge, potentially as an input to conflict flagging or salience.

## 9 · Documentation & OSS Polish
- Write `ARCHITECTURE.md` describing the storage format and workflow.
- Document the JSON store versioning rules in `STORAGE_FORMAT.md`.

## 10 · Future-Facing Stubs
- Leave TODOs for advanced health checks and split/merge logic.

## 11 · Cognitive-Inspired Enhancements: Active Memory & Encoding (Research / Future)
- **Refined Active Memory:**
    - Explore configurable capacity limits for active memory.
    - Define clear "refresh" mechanisms for items in active memory based on access/use.
    - Research sophisticated prioritization strategies beyond recency (e.g., salience, estimated utility).
    - *New:* Investigate a base system utility for managing a short-term 'working memory context' (e.g., a list of recently processed/retrieved item identifiers or summaries) that could be optionally used by any engine to inform its operations (e.g., to improve contextual understanding during compression or provide context for retrieval).
- **Deeper Semantic Ingestion:**
    - Actively use existing memory to contextualize new data during ingestion.
    - Research inference of explicit relationships (e.g., "extends," "contradicts") between new and existing memories.
- **User-Defined and Inferred Importance/Salience:**
            - Allow users to flag memories as "important," affecting encoding, persistence, and retrieval.
            - *New:* Explore mechanisms for the system to infer salience (e.g., based on novelty, connection to other important items, or explicit signals during interaction), similar to the `salience` concept in `neocortex_transfer.py`. This score could influence consolidation, decay, and retrieval weighting.
- **"Elaborative Encoding" Support:**
    - Allow storing user-provided elaborations/summaries linked to base memories.
    - *Advanced:* Research system-generated elaborations (e.g., auto-linking, relationship summaries).

## 12 · Cognitive-Inspired Enhancements: Consolidation & Dynamic Knowledge (Research / Future)
- **Active Memory Consolidation Processes (Background Task):**
    - Implement mechanisms for strengthening memories based on salience, access frequency, etc.
    - *Enhanced:* Generalize this to a base system concept of a 'memory maintenance phase' or 'consolidation hook' that engines can implement. This phase could be triggered periodically or manually.
    - *Advanced:* Research knowledge integration/reorganization:
        - Link refinement and indirect link discovery (e.g., inspired by `neocortex_transfer.py`'s linking).
        - Schema/cluster formation from interconnected memories (potential for `neocortex_transfer.py`).
        - Manage contradictions/updates during consolidation.
    - Design consolidation for "offline" execution during low system load.
- **Reconsolidation (Updating Memories on Retrieval):**
    - Research and implement mechanisms for memories to be updated/refined based on new information acquired during/after retrieval (making memories "malleable").
- **Explicit Memory States:**
    - Consider introducing explicit states for memories (e.g., "newly_encoded," "unconsolidated," "consolidated," "archived," "low_confidence") to guide system processes.
- *Advanced:* **Reconstructive Retrieval:**
    - Explore returning synthesized answers/summaries from multiple memories, not just single matches.


## 13. General System Properties (Cognitive Inspired)
- **Explicit Memory States:** Consider introducing explicit states for memories (e.g., "newly_encoded," "unconsolidated," "consolidated," "archived," "low_confidence") that influence how they are processed by different system components (decay, consolidation, retrieval). This was also listed under Consolidation, but has broader implications.
- **Standardized Rich Metadata for Memory Items:**
    - Define a more structured representation for individual memory items in the base system (e.g., a `BaseMemoryEntry` dataclass).
    - This could include fields like: `creation_timestamp`, `last_accessed_timestamp`, `access_count`, `strength_score` (generic, engine-defined meaning), `status_tags` (extensible list for states like 'new', 'archived'), and potentially `linked_item_ids`. This would provide a richer substrate for engines and system-level processes like maintenance or advanced retrieval.

## CLI Documentation & Usability
- Document global `--provider`, `--provider-url`, and `--provider-key` options in `docs/cli_reference.md` and `docs/USAGE.md`.
- Mention these options in the README quick-start section.
- Refactor CLI startup to delay loading heavy optional dependencies so `--help` runs quickly.
- Add cross-links to `docs/configuration.md` for environment variables used by CLI.
