# Sample Strategy Package for Compact Memory

This directory provides a template for packaging a custom compression strategy for use with the Compact Memory platform. Packaging allows your strategy to be more easily discovered, shared, and used within the Compact Memory ecosystem.

## Purpose

The main goal of this sample package is to illustrate:
1.  The necessary file structure for a strategy package.
2.  The metadata required to define your strategy.
3.  A basic implementation of a custom strategy.

You can adapt this example to package your own innovative compression strategies.

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

1.  **Copy this directory:** Start by copying this entire `sample_strategy_package` directory and renaming it to reflect your strategy's name.
2.  **Implement your strategy:**
    *   Modify `strategy.py` (or rename it and update `strategy_package.yaml` accordingly).
    *   Define your custom `CompressionStrategy` class, ensuring it has a unique `id` and implements the `compress` method.
3.  **Update metadata:**
    *   Edit `strategy_package.yaml` with your strategy's details (ID, class name, module, display name, version, authors, description).
4.  **Add dependencies:**
    *   If your strategy has dependencies, add them to `requirements.txt`.
5.  **Include examples:**
    *   Optionally, add example experiment configurations to the `experiments/` directory.

## Making Your Packaged Strategy Discoverable

Compact Memory can discover strategies packaged in this format if they are placed in a directory that the system scans. (The exact mechanism for registration or scanning might be detailed in the main Compact Memory documentation - e.g., by setting an environment variable or placing packages in a predefined location).

For local development and testing, you can often make your strategy available by ensuring the directory containing your package is in your `PYTHONPATH`, or by using local installation options if provided by Compact Memory's CLI or main setup.

Refer to the main Compact Memory documentation on "Sharing Strategies" ([`docs/SHARING_STRATEGIES.md`](../../docs/SHARING_STRATEGIES.md)) and "Developing Compression Strategies" ([`docs/DEVELOPING_COMPRESSION_STRATEGIES.md`](../../docs/DEVELOPING_COMPRESSION_STRATEGIES.md)) for more details on how strategies are loaded and best practices for development.
