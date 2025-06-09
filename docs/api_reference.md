# API Reference
This section provides a reference for the core public APIs of Compact Memory, particularly those essential for developing and using compression engines. For a more comprehensive and detailed API documentation, consider building the documentation using Sphinx with autodoc, which can generate it directly from the source code docstrings.
> **Note:** The content below is a manually curated list of key components. A full, auto-generated API reference (e.g., via Sphinx) is recommended for complete details and is planned for future updates.
## Core Engine Development APIs
The following classes and modules are fundamental when developing new compression engines.
### `compact_memory.engines.BaseCompressionEngine`
The abstract base class for all compression engines. Developers must subclass this to create new engines.
*   `Key methods: \`compress(self, text: str, budget: int, previous_compression_result: Optional[CompressedMemory] = None, **kwargs) -> CompressedMemory\``
*   `Key attributes: \`id\` (string identifier for the engine)`
### `compact_memory.engines.CompressedMemory`
A data class that holds the output of a compression operation.
*   `Key attributes: \`text: str\`, \`engine_id: Optional[str]\`, \`engine_config: Optional[Dict[str, Any]]\`, \`trace: Optional[CompressionTrace]\`, \`metadata: Optional[Dict[str, Any]]\``
### `compact_memory.engines.CompressionTrace`
A data class used to record the steps and metadata of a compression process. Essential for debugging and explainability.
*   `Key attributes: \`engine_name\`, \`engine_params\`, \`input_summary\`, \`steps\` (list of dicts), \`output_summary\`, \`processing_ms\`, \`final_compressed_object_preview\``
## Core Validation APIs
For evaluating engines, these components are key.
### `compact_memory.validation.metrics_abc.ValidationMetric`
The abstract base class for creating custom validation metrics to evaluate the quality or effectiveness of compressed memory or LLM responses based on it.
*   `Key methods: \`evaluate(...)\``
*   `Key attributes: \`metric_id\``
## Key Utility Modules
### `compact_memory.token_utils`
Provides helper functions for tokenization and token counting, useful within engines for budget management.
*   `Key functions: \`get_tokenizer(tokenizer_name_or_path)\`, \`token_count(tokenizer, text)\``
### `compact_memory.plugin_loader` and `compact_memory.registry`
These modules handle the discovery and registration of compression engines. Developers creating shareable packages will interact with the plugin system indirectly via entry points or package structure.
For detailed parameters, return types, and specific behaviors of these and other components, please refer to the docstrings within the Compact Memory source code. An auto-generated API reference would provide this in a navigable format.
