# Migration to the Engine Framework (v0.2.x and later)

Starting with version 0.2.0, Compact Memory has introduced a new "Engine" framework. This framework provides a more flexible and extensible way to manage different memory types, compression strategies, and their persistence. The most significant change for existing users is the replacement of the `MemoryContainer` class and the old `compact-memory memory` CLI commands.

## Key Changes

### 1. `MemoryContainer` Removed

The `MemoryContainer` class has been deprecated and its functionality is now handled by dedicated engine classes.

*   **Old:**
    ```python
    from compact_memory.memory_container import MemoryContainer
    container = MemoryContainer(embedding_dim=384) # Or loaded from path
    container.add_memory("Some text")
    results = container.query("text")
    container.save("my_memory")
    ```

*   **New (using a custom engine as an example):**
    ```python
    from compact_memory.engines import load_engine
    from compact_memory.vector_store import InMemoryVectorStore
    from compact_memory.embedding_pipeline import get_embedding_dim

    dim = get_embedding_dim()
    store = InMemoryVectorStore(embedding_dim=dim)
    engine = load_engine(store)
    engine.ingest("Some text")
    results = engine.recall("text")
    engine.save("my_engine_store")
    ```

### 2. Engine Persistence and Loading

Engines now have their own `save(path)` and `load(path)` methods for persistence. A generic `load_engine(path)` function is also available to load any engine type from a saved store.

*   **Saving an engine:**
    ```python
    engine.save("/path/to/my_engine_store")
    ```
    This creates an `engine_manifest.json` file within the specified directory, along with other data files required by the engine (for example, `memories.json` and `vectors.npy` for engines that store embeddings).

*   **Loading an engine:**
    ```python
    from compact_memory.engines import load_engine
    loaded_engine = load_engine("/path/to/my_engine_store")
    ```
    The `load_engine` function reads the manifest and instantiates the correct engine type.

### 3. Configuration Persistence

Engines now persist their configuration (parameters passed to `__init__` or set during their lifecycle) into the `engine_manifest.json` file. This means when an engine is loaded using `load_engine`, it will be restored with its previous configuration.

For details on how to manage and leverage configuration persistence in custom engines, please see the [Developing Compression Engines](./DEVELOPING_COMPRESSION_ENGINES.md) guide.

### 4. CLI Command Changes

The `compact-memory memory ...` command group has been replaced by `compact-memory engine ...`.

*   **Old:** `compact-memory memory init ...`
*   **New:** `compact-memory engine init --engine <engine_id> ...`

*   **Old:** `compact-memory memory stats ...`
*   **New:** `compact-memory engine stats ...`

*   **Old:** `compact-memory memory clear ...`
*   **New:** `compact_memory engine clear ...`

Refer to the [CLI Reference](./cli_reference.md) for updated command usage.

## Migration Steps

1.  **Update Imports:** Change imports from `MemoryContainer` to your chosen engine class.
2.  **Adapt Initialization:** If you were using `MemoryContainer`, update your code to initialize the new engine with an appropriate `VectorStore` instance.
3.  **Update Method Calls:**
    *   `container.add_memory(...)` on `MemoryContainer` maps to `engine.add_memory(...)` on your engine.
    *   `container.query(...)` maps to `engine.query(...)` on your engine.
    *   For more generic interaction, engines might implement `engine.ingest(...)` and `engine.recall(...)`.
4.  **Update Persistence:**
    *   Replace `container.save(path)` and `MemoryContainer.load(path)` with `engine.save(path)` and `load_engine(path)`.
5.  **Update CLI Scripts:** Modify any scripts using `compact-memory memory ...` to use the new `compact-memory engine ...` commands. Pay attention to changed options, especially for `init`.

If you have developed custom logic around `MemoryContainer`, you may need to adapt it to the new engine structure, potentially by creating your own `BaseCompressionEngine` subclass. See the [Developing Compression Engines](./DEVELOPING_COMPRESSION_ENGINES.md) guide for more information.
