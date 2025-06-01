# Developing Compression Strategies

Guidelines and examples for implementing new `CompressionStrategy` classes will be documented here.

## Designing Learnable Compression Strategies

Some compression approaches may incorporate trainable components—for example a small neural summarizer or a policy model that selects which passages to keep. When building such strategies:

1. Encapsulate the learnable model within your `CompressionStrategy` implementation so that the rest of the framework can treat it like any other strategy.
2. Provide `save_learnable_components(path)` and `load_learnable_components(path)` methods to persist and restore model state.
3. Consider how the experimentation framework might drive a simple training loop. For instance, a `ValidationMetric` could supply gradients or rewards that update your summarizer.
4. Document expected resources and dependencies so others can reproduce your results.

### Strategies for Conversational AI and Dynamic Contexts

The `ActiveMemoryManager` shows one pattern for maintaining dialogue context. Learnable strategies might extend this with models that predict relevance scores, or with reinforcement learning to optimize pruning decisions.
