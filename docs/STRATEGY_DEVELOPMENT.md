# BaseCompressionEngine Development Best Practices

## Writing a BaseCompressionEngine
- Inherit from `BaseCompressionEngine` and define a unique `id` string.
- Include metadata such as `display_name` and `version` when registering your engine.
- Keep implementations self‑contained and document any extra dependencies.

## Testing a BaseCompressionEngine
- Write unit tests for the `compress` method that cover normal and edge cases.
- Validate the returned `CompressionTrace` and ensure token counts are accurate.

## Sharing a BaseCompressionEngine
- Name your package using the `compact_memory_<name>_engine` pattern.
- Expose your engine via the `compact_memory.engines` entry point.
- Provide a `README.md` describing usage and installation requirements.
