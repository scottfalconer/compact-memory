# Troubleshooting

This guide helps you understand and resolve common issues you might encounter while using the Compact Memory library and CLI.

## Common Errors and Custom Exceptions

Compact Memory uses a hierarchy of custom exceptions to provide more specific error information. When an issue occurs, especially via the CLI, you'll often see these error types mentioned. Understanding them can help pinpoint the problem.

Here's a list of primary custom exceptions and what they generally signify:

*   **`CompactMemoryError`**: This is the base class for most library-specific errors. If you see this, it's an issue originating from the Compact Memory codebase.

    *   **`EngineError`**: A subclass of `CompactMemoryError`, this is the base for errors specifically related to compression engines.
        *   **`EngineLoadError`**: Raised when an engine fails to load from disk. This could be due to:
            *   Missing or corrupted `engine_manifest.json` or `entries.json`.
            *   Errors decoding JSON data from these files.
            *   Failure to load dynamically specified `embedding_fn_path` or `preprocess_fn_path` (e.g., module not found, function not in module).
            *   Issues loading the associated `VectorStore`.
            *   The CLI will typically report messages like "Error loading engine: Manifest or essential file not found..." or "Failed to load embedding_fn...".
        *   **`EngineSaveError`**: Raised when an engine fails to save its state to disk. This might happen due to:
            *   File permission issues.
            *   Errors during serialization of engine data.
            *   Failures in the `VectorStore`'s own save process.
            *   The CLI will report messages like "Engine Save Error: Failed to save engine to <path>...".
        *   **`EmbeddingDimensionMismatchError`**: (Also an `EngineError`) Raised when there's a mismatch in embedding dimensions. This commonly occurs if:
            *   An engine is initialized or loaded with an `embedding_dim` in its configuration that conflicts with the dimension of the actual embedding model being used (especially the default one).
            *   A pre-existing vector store has a different embedding dimension than what the engine or its current configuration expects.
            *   CLI message: "Configuration Error: Embedding dimension mismatch..."

    *   **`VectorStoreError`**: A subclass of `CompactMemoryError`, this is the base for errors related to vector store operations.
        *   **`IndexRebuildError`**: Raised specifically when rebuilding a vector store's search index fails. This is often related to issues persisting the rebuilt index for stores like `PersistentFaissVectorStore`.
        *   Other `VectorStoreError` instances can be raised during `save` or `load` operations of a vector store if, for example, specific files like `embeddings.npy` or `index.faiss` are missing or corrupted within the vector store's own data directory.
        *   CLI messages might include "Error loading vector store: File not found..." or "Failed to persist rebuilt FAISS index...".

    *   **`ConfigurationError`**: Raised for general configuration problems not covered by more specific engine or vector store errors. This can include:
        *   Failure to determine a critical configuration value (e.g., `embedding_dim` if not specified and not inferable).
        *   Issues creating a component (like a `VectorStore` or `Chunker`) due to invalid configuration parameters.
        *   Problems with dynamically loading functions specified by path in `EngineConfig` if these occur outside the direct load path of an engine.
        *   CLI message: "Configuration Error: Cannot determine embedding dimension..."

## Interpreting CLI Error Messages

When using the `compact-memory` CLI, errors derived from `CompactMemoryError` will generally be caught and presented in a user-friendly way, often prefixed with "Error:", "Configuration Error:", "Engine Load Error:", etc.

*   **Read the message carefully:** It will usually contain the specific path or setting that caused the issue.
*   **Check file paths:** Ensure that paths provided via `--memory-path` or other options are correct and that the necessary files/directories exist and have correct permissions.
*   **Verify configuration:** If an error points to a configuration issue (e.g., `EmbeddingDimensionMismatchError`, `ConfigurationError`), review your `EngineConfig` or how the engine was initialized.
*   **Enable verbose logging:** For more detailed insight, use the global CLI options `--verbose` (or `-V`) and `--log-file <path_to_log>`. This will provide DEBUG level logs which can show the sequence of operations leading to the error. See the "Logging" section in `docs/CONFIGURATION.md` for more details.

## Common Scenarios and Fixes

*   **`EngineLoadError: Manifest or essential file not found`**:
    *   Ensure the path to the engine store is correct.
    *   Verify that `engine_manifest.json` and `entries.json` exist in the specified directory.
    *   Check that the `vector_store_data` subdirectory also exists and contains the necessary files for your vector store type.

*   **`EngineLoadError: Failed to load embedding_fn from path ...`**:
    *   Make sure the Python module and function specified in `embedding_fn_path` (in `engine_manifest.json` -> `config`) are available in your Python environment (PYTHONPATH).
    *   Check for typos in the path string.

*   **`EmbeddingDimensionMismatchError`**:
    *   If you're using a custom embedding model (via `embedding_fn_path`), ensure `embedding_dim` in your `EngineConfig` matches your model's output dimension.
    *   If using the default embedding model, this error is less common unless an old store is being loaded with a new default model of a different size.

*   **`IndexRebuildError` during `engine rebuild-index`**:
    *   This usually indicates a problem saving the rebuilt index for persistent stores (like `PersistentFaissVectorStore`). Check file permissions for the `vector_store_data/index.faiss` file within your engine directory.

If you encounter persistent issues, referring to the detailed logs (using `--verbose` and `--log-file`) is often the best next step.
