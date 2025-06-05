# Sharing and Discovering Compression Strategies

Compact Memory's plugin system enables developers to create, package, and share their custom compression strategies, and for users to easily discover and integrate them. This document outlines the plugin architecture, how to package a strategy, and how to use shared strategies.

## Plugin Architecture

Compact Memory discovers available compression strategies through two primary mechanisms:

1.  **Python Package Entry Points:**
    *   This is the standard way to make strategies discoverable when they are part of an installable Python package (e.g., distributed via PyPI).
    *   Your package's `pyproject.toml` (or `setup.py`/`setup.cfg`) should declare an entry point under the `compact_memory.strategies` group.
    *   **Example (`pyproject.toml`):**
        ```toml
        [project.entry-points."compact_memory.strategies"]
        your_strategy_id = "your_package.module_name:YourStrategyClassName"
        ```
        Where:
        *   `your_strategy_id` is the unique string ID for your strategy (must match the `id` attribute in your strategy class).
        *   `your_package.module_name` is the importable path to the Python module containing your strategy class.
        *   `YourStrategyClassName` is the name of your strategy class that inherits from `CompressionStrategy`.
    *   When a Python package with such an entry point is installed in the environment, Compact Memory will automatically detect and register the strategy.

2.  **Local Plugin Directories:**
    *   For development, testing, or simpler distribution without creating a full Python package, Compact Memory can load strategies from specific local directories.
    *   These directories must contain valid "strategy packages" (see below).
    *   Compact Memory checks the path specified by the `COMPACT_MEMORY_PLUGINS_PATH` environment variable. This variable can contain multiple paths, separated by the OS's standard path separator (e.g., `:` for Linux/macOS, `;` for Windows).
    *   It also checks a default user-specific plugin directory (e.g., `~/.local/share/compact_memory/plugins` on Linux, or similar paths on other OSes, determined by the `platformdirs` library).
    *   Strategies found in these directories are registered. If multiple strategies with the same ID are found, a precedence order applies (typically, local directory plugins might override entry point plugins, and built-in strategies have the highest precedence unless overridden by a plugin specifically designed to do so). Use `compact-memory dev list-strategies` to see which strategy is active and its source.

**Security Warning:** Loading plugins, especially from local directories or third-party packages, involves executing arbitrary Python code. Only install or load plugins from sources you trust.

## Creating a Shareable Strategy Package

A "strategy package" is a directory with a specific structure and a manifest file (`strategy_package.yaml`) that describes the custom strategy. This format is used for local plugin directories and can also be the basis for a full Python package.

### Package Structure

A typical strategy package directory looks like this:

```
my_custom_strategy_package/
├── strategy.py                      # Your CompressionStrategy implementation
├── strategy_package.yaml            # Manifest file (required)
├── README.md                        # Documentation for your strategy
├── requirements.txt                 # (Optional) Python dependencies
└── experiments/                     # (Optional) Example experiment configs
    └── example_experiment.yaml
```

### The `strategy_package.yaml` Manifest

This YAML file is crucial for Compact Memory to understand and load your strategy.

**Required Fields:**

*   `package_format_version` (string): Version of the package format (e.g., "1.0").
*   `strategy_id` (string): The unique identifier for your strategy. This **must** match the `id` class attribute in your `CompressionStrategy` implementation.
*   `strategy_class_name` (string): The name of your Python class that implements the strategy (e.g., `MyCoolStrategy`).
*   `strategy_module` (string): The name of the Python module file (without the `.py` extension) within your package that contains the strategy class (e.g., `strategy` if your file is `strategy.py`).
*   `display_name` (string): A user-friendly name for your strategy that will appear in listings.
*   `version` (string): The version number of your strategy package (e.g., "0.1.0").
*   `authors` (list of strings): Names or contact info of the authors.
*   `description` (string): A brief description of what your strategy does.

**Optional Fields:**

*   `dependencies` (list of strings): A list of Python package dependencies (can also be in `requirements.txt`).
*   `default_experiments` (list of dicts): Can define default experiments for `compact-memory dev run-package-experiment`.

**Example (`strategy_package.yaml`):**

```yaml
package_format_version: "1.0"
strategy_id: my_extractive_summarizer
strategy_class_name: MyExtractiveSummarizerStrategy
strategy_module: custom_summarizer_module  # Corresponds to custom_summarizer_module.py
display_name: My Awesome Extractive Summarizer
version: "1.0.2"
authors:
  - "Researcher X <rx@example.com>"
description: "An extractive summarization strategy using sentence embeddings and clustering."
dependencies:
  - "scikit-learn>=1.0"
  - "numpy"
```

Refer to the example package at `examples/sample_strategy_package/` for a working template.

### Automating Package Creation with `dev create-strategy-package`

To help you get started quickly, Compact Memory provides a CLI command to generate a template for a new strategy package:

```bash
compact-memory dev create-strategy-package --name compact_memory_my_strategy
```

This command will:
1.  Create a directory named `compact_memory_my_strategy` (or as specified by `--path`).
2.  Populate it with template files:
    *   `strategy.py` (with a skeleton `CompressionStrategy` class named `MyStrategy`).
    *   `strategy_package.yaml` (pre-filled with `strategy_id: YourStrategyName`, `strategy_class_name: MyStrategy`, etc.).
    *   A basic `README.md`.
    *   An empty `requirements.txt`.
    *   An `experiments/` directory with an `example.yaml`.

You then need to:
1.  Edit `strategy.py` to implement your compression logic and set the correct `id` in the class.
2.  Update `strategy_package.yaml` to ensure `strategy_id`, `strategy_class_name`, `display_name`, `authors`, and `description` are accurate for your strategy.
3.  Add any dependencies to `requirements.txt`.
4.  Write documentation in `README.md`.

### Validating Your Package

Before sharing, you can validate your package structure and manifest:
```bash
compact-memory dev validate-strategy-package path/to/your_strategy_package_directory
```
This will check for required files, fields in the manifest, and basic loadability of the strategy class.

## Finding, Installing, and Using Shared Strategies

1.  **Finding Strategies:**
    *   Community strategies might be listed in a dedicated section of the Compact Memory documentation (like `docs/community_strategies.md` - work in progress).
    *   They might be hosted on PyPI or code repositories like GitHub.

2.  **Installing Strategies:**
    *   **As Python Packages:** If distributed via PyPI, install using pip:
        ```bash
        pip install name-of-strategy-package
        ```
        If it's a wheel file or from a Git repository:
        ```bash
        pip install path/to/strategy.whl
        pip install git+https://github.com/user/repo.git
        ```
    *   **As Directory Packages:** Download or clone the strategy package directory and place it into one of the locations scanned by Compact Memory (see "Local Plugin Directories" above).

3.  **Listing Available Strategies:**
    To see all strategies Compact Memory has discovered (built-in, from entry points, and from local plugin directories), use:
    ```bash
    compact-memory dev list-strategies
    ```
    This command will show the strategy ID, display name, version, and source, which helps verify if your shared strategy is correctly loaded.

4.  **Using Shared Strategies:**
    Once a strategy is installed and discoverable, you can use it just like a built-in one by referencing its `strategy_id`:
    *   **CLI:**
        ```bash
        compact-memory compress --file input.txt --strategy <shared_strategy_id> --budget <value>
        ```
    *   **Python API:**
        ```python
        from compact_memory.compression import get_compression_strategy

        strategy_instance = get_compression_strategy("<shared_strategy_id>")()
        compressed_mem, trace = strategy_instance.compress(text, budget)
        # ...
        ```

By following these guidelines, you can contribute to and benefit from a growing ecosystem of powerful context compression tools for LLMs.
