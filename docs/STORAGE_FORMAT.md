# Storage Format

This document describes how a Compact Memory memory store is kept on disk.  A memory store is a directory containing several JSON and NPY files along with a small YAML metadata file.  The layout is intentionally simple so that the contents can be inspected or backed up with regular tools.

```
memory/
├── meta.yaml
├── belief_prototypes.json
├── prototype_vectors.npy
├── raw_memories.jsonl
├── evidence.jsonl        # log of prototype ↔ memory links
└── conflicts.jsonl       # optional contradictions log
```

## `meta.yaml`

`meta.yaml` holds global information about the store:

```yaml
version: 1                     # storage schema version
embedding_model: all-MiniLM-L6-v2
embedding_dim: 384            # vector dimension
normalized: true              # embeddings must be unit vectors
created_at: "2024-01-01T00:00:00Z"
updated_at: "2024-01-01T00:00:00Z"
```

The `version` field defines the storage schema.  The current code understands
version `1`.  Should a future release change the file layout, this value will be
incremented and migration logic will be added.  Tools loading a memory store should
check the `version` field before attempting to read the other files.

## Prototype files

`belief_prototypes.json` stores the prototype metadata without vectors.  Each
entry is a JSON object matching the `BeliefPrototype` model.  The associated
vectors are stored row‑for‑row in `prototype_vectors.npy` which is loaded with
NumPy.

## Memories

Individual chunks of text are appended to `raw_memories.jsonl`.  Each line is a
`RawMemory` JSON object containing the text, its hash, optional embedding and the
prototype it belongs to.

## Logs

During ingestion the agent writes two auxiliary logs:

- `evidence.jsonl` records which memories contributed to each prototype.  It can
  be analysed offline to track provenance or compute statistics.
- `conflicts.jsonl` stores potential contradictions flagged by the heuristic
  checker.  These rows are meant for human review and can be deleted without
  affecting the core store.

The storage format is purposely lightweight and versioned so that different
backends or migration tools can be implemented without breaking existing data.
