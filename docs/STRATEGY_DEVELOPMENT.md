# CompressionStrategy Development Best Practices

## Writing a CompressionStrategy
- Inherit from `CompressionStrategy` and define a unique `id` string.
- Include metadata such as `display_name` and `version` when registering your strategy.
- Keep implementations self‑contained and document any extra dependencies.

## Testing a CompressionStrategy
- Write unit tests for the `compress` method that cover normal and edge cases.
- Validate the returned `CompressionTrace` and ensure token counts are accurate.

## Sharing a CompressionStrategy
- Name your package using the `compact_memory_<name>_strategy` pattern.
- Expose your strategy via the `compact_memory.strategies` entry point.
- Provide a `README.md` describing usage and installation requirements.
