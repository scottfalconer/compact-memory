# Developing Compression Strategies

This document is a practical guide for developers on how to implement new `CompressionStrategy` modules within the Gist Memory framework. While `docs/COMPRESSION_STRATEGIES.md` covers the conceptual and theoretical aspects of designing such strategies (the "what" and "why"), this document focuses on the "how-to" – the specific interfaces, methods, and considerations for writing the code.

## Designing Learnable Compression Strategies

Some compression approaches may incorporate trainable components—for example a small neural summarizer or a policy model that selects which passages to keep. When building such strategies:

1. Encapsulate the learnable model within your `CompressionStrategy` implementation so that the rest of the framework can treat it like any other strategy.
2. Provide `save_learnable_components(path)` and `load_learnable_components(path)` methods to persist and restore model state.
3. Consider how the experimentation framework might drive a simple training loop. For instance, a `ValidationMetric` could supply gradients or rewards that update your summarizer.
4. Document expected resources and dependencies so others can reproduce your results.

### Strategies for Conversational AI and Dynamic Contexts

The `ActiveMemoryManager` shows one pattern for maintaining dialogue context. Learnable strategies might extend this with models that predict relevance scores, or with reinforcement learning to optimize pruning decisions.

## Creating Informative Compression Traces

Every `CompressionStrategy` should return a `CompressionTrace` detailing the
steps performed. Use the standard vocabulary from
`docs/EXPLAINABLE_COMPRESSION.md` for the `type` field and include contextual
information in a `details` dictionary. A minimal example:

```python
trace.steps.append({
    "type": "prune_history_turn",
    "details": {
        "turn_id": "abc-123",
        "text_preview": "User: Yes, that sounds right...",
        "reason_for_action": "lowest_retention_score",
        "retention_score": 0.15,
    },
})
```

These rich traces make strategies easier to debug and analyse. When designing a
new strategy, think about what decisions are being made and record them as steps
in the trace.
