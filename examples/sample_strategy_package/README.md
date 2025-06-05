# Sample Engine Package for Compact Memory

This directory provides a template for packaging a custom compression engine for use with the Compact Memory platform. Packaging allows your engine to be more easily discovered, shared, and used within the Compact Memory ecosystem.

## Purpose

The main goal of this sample package is to illustrate:
1.  The necessary file structure for a engine package.
2.  The metadata required to define your engine.
3.  A basic implementation of a custom engine.

You can adapt this example to package your own innovative compression engines.
It serves as a foundational template that developers can copy and modify to quickly bootstrap their own engine development.

## File Structure

A typical engine package includes the following:

*   **`engine.py`**:
    *   This is where you implement your custom `CompressionEngine`.
    *   In this example, `engine.py` contains `SampleEngine`, a trivial engine that returns the input text unchanged.
    *   Your engine should inherit from `compact_memory.compression.engines_abc.CompressionEngine` and implement the `compress` method. It must also have a unique `id` attribute.

*   **`engine_package.yaml`**:
    *   This metadata file describes your engine package. It includes:
        *   `package_format_version`: Version of the packaging format.
        *   `engine_id`: The unique ID of your engine (should match the `id` in your engine class).
        *   `engine_class_name`: The class name of your engine (e.g., `SampleEngine`).
        *   `engine_module`: The Python module where your engine class is defined (e.g., `engine` if your file is `engine.py`).
        *   `display_name`: A user-friendly name for your engine.
        *   `version`: Version of your engine package.
        *   `authors`: List of authors.
        *   `description`: A brief description of what your engine does.
    *   Refer to the example `engine_package.yaml` in this directory.

*   **`requirements.txt` (Optional)**:
    *   If your engine has specific Python dependencies, list them here.
    *   These can be installed by users of your engine package.

*   **`experiments/` (Optional)**:
    *   You can include example experiment configuration files (`.yaml`) that demonstrate how to use your engine.
    *   See `experiments/example.yaml` for a basic structure.

## Adapting This Example

1. **Copy and Rename:** Duplicate this entire `sample_engine_package` directory. Rename the copied directory to something descriptive for your new engine (e.g., `my_awesome_engine_package`).
2. **Implement Your Engine Logic:** Open the `.py` file (e.g., `engine.py`, or rename it if you prefer). This is where you'll write the core logic of your compression algorithm.
    *   Define your custom class inheriting from `CompressionEngine`.
    *   Ensure your class has a unique static string attribute named `id` (e.g., `id = "my_awesome_engine"`). This ID is how the system will recognize and load your engine.
    *   Implement the `compress` method. This method takes the input text/chunks and a token budget and should return a `CompressedMemory` object and a `CompressionTrace` object.
3. **Update Metadata (`engine_package.yaml`):** Edit the `engine_package.yaml` file to reflect your engine's details. Critically, ensure `engine_id` matches the `id` attribute in your engine class, and `engine_module` and `engine_class_name` correctly point to your Python file and class.
4. **List Dependencies (`requirements.txt`):** If your engine relies on external Python libraries, add them to the `requirements.txt` file, one per line (e.g., `numpy>=1.20`).
5. **Add Examples (`experiments/`):** Optionally, create or modify YAML configuration files in the `experiments/` directory to showcase how to run experiments with your engine.

## Making Your Packaged Engine Discoverable

Compact Memory can discover engines packaged in this format if they are placed in a directory that the system scans. (The exact mechanism for registration or scanning might be detailed in the main Compact Memory documentation - e.g., by setting an environment variable or placing packages in a predefined location).

For local development and testing, you can often make your engine available by ensuring the directory containing your package is in your `PYTHONPATH`, or by using local installation options if provided by Compact Memory's CLI or main setup.

Refer to the main Compact Memory documentation on "Sharing Strategies" ([`docs/SHARING_STRATEGIES.md`](../../docs/SHARING_STRATEGIES.md)) and "Developing Compression Strategies" ([`docs/DEVELOPING_COMPRESSION_STRATEGIES.md`](../../docs/DEVELOPING_COMPRESSION_STRATEGIES.md)) for more details on how engines are loaded and best practices for development.

## Key Components Explained

*   **`CompressionEngine` (from `compact_memory.compression.engines_abc`):** This is the abstract base class that all engines must inherit from. It defines the interface that the Compact Memory framework expects.
*   **`id` attribute (in your engine class):** A unique string that identifies your engine. This is crucial for the framework to find and load your engine.
*   **`compress` method (in your engine class):** The heart of your engine. It receives text and a budget and must return `CompressedMemory` (containing the compressed text) and `CompressionTrace` (for logging and debugging).
*   **`engine_package.yaml`:** The manifest file. It tells the Compact Memory framework how to find and interpret your engine.
