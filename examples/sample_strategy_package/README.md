# Sample Strategy Package for Compact Memory

This directory provides a template for packaging a custom compression strategy for use with the Compact Memory platform. Packaging allows your strategy to be more easily discovered, shared, and used within the Compact Memory ecosystem.

## Purpose

The main goal of this sample package is to illustrate:
1.  The necessary file structure for a strategy package.
2.  The metadata required to define your strategy.
3.  A basic implementation of a custom strategy.

You can adapt this example to package your own innovative compression strategies.
It serves as a foundational template that developers can copy and modify to quickly bootstrap their own strategy development.

## File Structure

A typical strategy package includes the following:

*   **`strategy.py`**:
    *   This is where you implement your custom `CompressionStrategy`.
    *   In this example, `strategy.py` contains `SampleStrategy`, a trivial strategy that returns the input text unchanged.
    *   Your strategy should inherit from `compact_memory.compression.strategies_abc.CompressionStrategy` and implement the `compress` method. It must also have a unique `id` attribute.

*   **`strategy_package.yaml`**:
    *   This metadata file describes your strategy package. It includes:
        *   `package_format_version`: Version of the packaging format.
        *   `strategy_id`: The unique ID of your strategy (should match the `id` in your strategy class).
        *   `strategy_class_name`: The class name of your strategy (e.g., `SampleStrategy`).
        *   `strategy_module`: The Python module where your strategy class is defined (e.g., `strategy` if your file is `strategy.py`).
        *   `display_name`: A user-friendly name for your strategy.
        *   `version`: Version of your strategy package.
        *   `authors`: List of authors.
        *   `description`: A brief description of what your strategy does.
    *   Refer to the example `strategy_package.yaml` in this directory.

*   **`requirements.txt` (Optional)**:
    *   If your strategy has specific Python dependencies, list them here.
    *   These can be installed by users of your strategy package.

*   **`experiments/` (Optional)**:
    *   You can include example experiment configuration files (`.yaml`) that demonstrate how to use your strategy.
    *   See `experiments/example.yaml` for a basic structure.

## Adapting This Example

1. **Copy and Rename:** Duplicate this entire `sample_strategy_package` directory. Rename the copied directory to something descriptive for your new strategy (e.g., `my_awesome_strategy_package`).
2. **Implement Your Strategy Logic:** Open the `.py` file (e.g., `strategy.py`, or rename it if you prefer). This is where you'll write the core logic of your compression algorithm.
    *   Define your custom class inheriting from `CompressionStrategy`.
    *   Ensure your class has a unique static string attribute named `id` (e.g., `id = "my_awesome_strategy"`). This ID is how the system will recognize and load your strategy.
    *   Implement the `compress` method. This method takes the input text/chunks and a token budget and should return a `CompressedMemory` object and a `CompressionTrace` object.
3. **Update Metadata (`strategy_package.yaml`):** Edit the `strategy_package.yaml` file to reflect your strategy's details. Critically, ensure `strategy_id` matches the `id` attribute in your strategy class, and `strategy_module` and `strategy_class_name` correctly point to your Python file and class.
4. **List Dependencies (`requirements.txt`):** If your strategy relies on external Python libraries, add them to the `requirements.txt` file, one per line (e.g., `numpy>=1.20`).
5. **Add Examples (`experiments/`):** Optionally, create or modify YAML configuration files in the `experiments/` directory to showcase how to run experiments with your strategy.

## Making Your Packaged Strategy Discoverable

Compact Memory can discover strategies packaged in this format if they are placed in a directory that the system scans. (The exact mechanism for registration or scanning might be detailed in the main Compact Memory documentation - e.g., by setting an environment variable or placing packages in a predefined location).

For local development and testing, you can often make your strategy available by ensuring the directory containing your package is in your `PYTHONPATH`, or by using local installation options if provided by Compact Memory's CLI or main setup.

Refer to the main Compact Memory documentation on "Sharing Strategies" ([`docs/SHARING_STRATEGIES.md`](../../docs/SHARING_STRATEGIES.md)) and "Developing Compression Strategies" ([`docs/DEVELOPING_COMPRESSION_STRATEGIES.md`](../../docs/DEVELOPING_COMPRESSION_STRATEGIES.md)) for more details on how strategies are loaded and best practices for development.

## Key Components Explained

*   **`CompressionStrategy` (from `compact_memory.compression.strategies_abc`):** This is the abstract base class that all strategies must inherit from. It defines the interface that the Compact Memory framework expects.
*   **`id` attribute (in your strategy class):** A unique string that identifies your strategy. This is crucial for the framework to find and load your strategy.
*   **`compress` method (in your strategy class):** The heart of your strategy. It receives text and a budget and must return `CompressedMemory` (containing the compressed text) and `CompressionTrace` (for logging and debugging).
*   **`strategy_package.yaml`:** The manifest file. It tells the Compact Memory framework how to find and interpret your strategy.
