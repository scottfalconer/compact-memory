# Explainable Compression and Trace Visualization

Compact Memory aims to make compression engines transparent and debuggable. A
`CompressionTrace` records the internal decisions of an engine so that
experiments can be analysed after the fact. The following guidelines describe how
to produce informative traces and inspect them.

## Trace Step Vocabulary

Each step in ``CompressionTrace.steps`` should include a ``type`` string and a
``details`` dictionary. The recommended ``type`` values are:

- ``chunk_text`` – initial splitting of text into chunks
- ``filter_item`` – removing an item such as a chunk or sentence
- ``select_item`` – choosing an item to keep
- ``rank_items`` – ordering items by a criterion
- ``summarize_item`` – generating a summary
- ``transform_item`` – modifying an item
- ``merge_items`` – combining multiple items
- ``prune_history_turn`` – removing a conversation turn from active memory
- ``retrieve_ltm_candidate`` – fetching a candidate from long‑term memory
- ``truncate_content`` – shortening content to fit a budget
- ``score_item`` – assigning a relevance or importance score

This list is extensible; engines may introduce additional types when needed.

### Suggested ``details`` keys

Provide structured context for each step using keys such as:

- identifiers: ``item_id``, ``chunk_id``, ``turn_id``
- content snippets: ``original_text_preview``, ``processed_text_preview``
- quantitative data: ``original_token_count``, ``processed_token_count``,
  ``score_value``
- qualitative reasoning: ``reason_for_action``, ``criterion_used``
- model or method information: ``model_used``, ``parameters_used``

Consistent fields help downstream tooling display and compare traces.

## Inspecting Traces

Use ``compact-memory trace inspect <trace.json>`` to print a human-readable summary
of a saved ``CompressionTrace``. A ``--type`` option filters steps by type.

```
$ compact-memory trace inspect trace.json --type filter_item
```

The command prints the engine name, then lists the matching steps with a short
preview of each ``details`` dictionary. This makes it easy to understand why
items were kept or removed during compression.

## Visualisation (Experimental)

Traces may also include saliency information so that selected text can be
highlighted. The optional ``compact-memory trace visualize`` command renders an HTML
file showing which parts of the original input influenced the output. This feature
is experimental and only supported by certain engines.
