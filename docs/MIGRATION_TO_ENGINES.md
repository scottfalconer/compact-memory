# Migration Guide: Strategies and Containers to Engines

Compact Memory 1.0 replaces the old *compression strategy* and *memory container* concepts with a unified **compression engine** API. Existing code that relied on `CompressionStrategy` subclasses or container management utilities should update to the new engine classes.

## Key Changes

- `CompressionStrategy` -> `BaseCompressionEngine`
- `register_compression_strategy` -> `register_compression_engine`
- Strategy packages now use `engine.py` and `engine_package.yaml`
- CLI options referencing `--strategy` have been renamed to `--engine`
- Memory containers are loaded and saved directly through engine methods

Custom strategies can usually be migrated by renaming the class to subclass `BaseCompressionEngine` and adjusting method signatures. Engine registration and discovery follow the same plugin mechanism as before, but under the `compact_memory.engines` entry point group.

See updated examples and documentation for concrete usage.
