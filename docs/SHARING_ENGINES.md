# Sharing and Discovering Compression Engines

Compact Memory's plugin system enables developers to create, package, and share their custom compression engines, and for users to easily discover and integrate them. This document outlines the plugin architecture, how to package a engine, and how to use shared engines.

## Plugin Architecture

Compact Memory discovers available compression engines through two primary mechanisms:

1.  **Python Package Entry Points:**
    *   This is the standard way to make engines discoverable when they are part of an installable Python package (e.g., distributed via PyPI).
    *   Your package's `pyproject.toml` (or `setup.py`/`setup.cfg`) should declare an entry point under the `compact_memory.engines` group.
    *   **Example (`pyproject.toml`):**
        ```toml
        [project.entry-points."compact_memory.engines"]
        your_engine_id = "your_package.module_name:YourEngineClassName"
        ```
        Where:
        *   `your_engine_id` is the unique string ID for your engine (must match the `id` attribute in your engine class).
        *   `your_package.module_name` is the importable path to the Python module containing your engine class.
        *   `YourEngineClassName` is the name of your engine class that inherits from `BaseCompressionEngine`.
    *   When a Python package with such an entry point is installed in the environment, Compact Memory will automatically detect and register the engine.

2.  **Local Plugin Directories:**
    *   For development, testing, or simpler distribution without creating a full Python package, Compact Memory can load engines from specific local directories.
    *   These directories must contain valid "engine packages" (see below).
*   Compact Memory checks the path specified by the `COMPACT_MEMORY_ENGINES_PATH` environment variable. This variable can contain multiple paths, separated by the OS's standard path separator (e.g., `:` for Linux/macOS, `;` for Windows).
    *   It also checks a default user-specific plugin directory (e.g., `~/.local/share/compact_memory/plugins` on Linux, or similar paths on other OSes, determined by the `platformdirs` library).
    *   Engines found in these directories are registered. If multiple engines with the same ID are found, a precedence order applies (typically, local directory plugins might override entry point plugins, and built-in engines have the highest precedence unless overridden by a plugin specifically designed to do so). Use `compact-memory dev list-engines` to see which engine is active and its source.

**Security Warning:** Loading plugins, especially from local directories or third-party packages, involves executing arbitrary Python code. Only install or load plugins from sources you trust.

## Creating a Shareable Engine Package

A "engine package" is a directory with a specific structure and a manifest file (`engine_package.yaml`) that describes the custom engine. This format is used for local plugin directories and can also be the basis for a full Python package.

### Package Structure

A typical engine package directory looks like this:

```
my_custom_engine_package/
├── engine.py                      # Your BaseCompressionEngine implementation
├── engine_package.yaml            # Manifest file (required)
├── README.md                        # Documentation for your engine
├── requirements.txt                 # (Optional) Python dependencies
└── experiments/                     # (Optional) Example experiment configs
    └── example_experiment.yaml
```

### The `engine_package.yaml` Manifest

This YAML file is crucial for Compact Memory to understand and load your engine.

**Required Fields:**

*   `package_format_version` (string): Version of the package format (e.g., "1.0").
*   `engine_id` (string): The unique identifier for your engine. This **must** match the `id` class attribute in your `BaseCompressionEngine` implementation.
*   `engine_class_name` (string): The name of your Python class that implements the engine (e.g., `MyCoolEngine`).
*   `engine_module` (string): The name of the Python module file (without the `.py` extension) within your package that contains the engine class (e.g., `engine` if your file is `engine.py`).
*   `display_name` (string): A user-friendly name for your engine that will appear in listings.
*   `version` (string): The version number of your engine package (e.g., "0.1.0").
*   `authors` (list of strings): Names or contact info of the authors.
*   `description` (string): A brief description of what your engine does.

**Optional Fields:**

*   `dependencies` (list of strings): A list of Python package dependencies (can also be in `requirements.txt`).
*   `default_experiments` (list of dicts): Can define default experiments for `compact-memory dev run-package-experiment`.

**Example (`engine_package.yaml`):**

```yaml
package_format_version: "1.0"
engine_id: my_extractive_summarizer
engine_class_name: MyExtractiveSummarizerEngine
engine_module: custom_summarizer_module  # Corresponds to custom_summarizer_module.py
display_name: My Awesome Extractive Summarizer
version: "1.0.2"
authors:
  - "Researcher X <rx@example.com>"
description: "An extractive summarization engine using sentence embeddings and clustering."
dependencies:
  - "scikit-learn>=1.0"
  - "numpy"
```

Refer to the example package at `examples/sample_engine_package/` for a working template.

### Automating Package Creation with `dev create-engine-package`

To help you get started quickly, Compact Memory provides a CLI command to generate a template for a new engine package:

```bash
compact-memory dev create-engine-package --name compact_memory_my_engine
```

This command will:
1.  Create a directory named `compact_memory_my_engine` (or as specified by `--path`).
2.  Populate it with template files:
    *   `engine.py` (with a skeleton `BaseCompressionEngine` class named `MyEngine`).
    *   `engine_package.yaml` (pre-filled with `engine_id: YourEngineName`, `engine_class_name: MyEngine`, etc.).
    *   A basic `README.md`.
    *   An empty `requirements.txt`.
    *   An `experiments/` directory with an `example.yaml`.

You then need to:
1.  Edit `engine.py` to implement your compression logic and set the correct `id` in the class.
2.  Update `engine_package.yaml` to ensure `engine_id`, `engine_class_name`, `display_name`, `authors`, and `description` are accurate for your engine.
3.  Add any dependencies to `requirements.txt`.
4.  Write documentation in `README.md`.

### Validating Your Package

Before sharing, you can validate your package structure and manifest:
```bash
compact-memory dev validate-engine-package path/to/your_engine_package_directory
```
This will check for required files, fields in the manifest, and basic loadability of the engine class.

## Finding, Installing, and Using Shared Engines

1.  **Finding Engines:**
    *   Community engines might be listed in a dedicated section of the Compact Memory documentation (like `docs/community_engines.md` - work in progress).
    *   They might be hosted on PyPI or code repositories like GitHub.

2.  **Installing Engines:**
    *   **As Python Packages:** If distributed via PyPI, install using pip:
        ```bash
        pip install name-of-engine-package
        ```
        If it's a wheel file or from a Git repository:
        ```bash
        pip install path/to/engine.whl
        pip install git+https://github.com/user/repo.git
        ```
    *   **As Directory Packages:** Download or clone the engine package directory and place it into one of the locations scanned by Compact Memory (see "Local Plugin Directories" above).

3.  **Listing Available Engines:**
    To see all engines Compact Memory has discovered (built-in, from entry points, and from local plugin directories), use:
    ```bash
    compact-memory dev list-engines
    ```
    This command will show the engine ID, display name, version, and source, which helps verify if your shared engine is correctly loaded.

4.  **Using Shared Engines:**
    Once a engine is installed and discoverable, you can use it just like a built-in one by referencing its `engine_id`:
    *   **CLI:**
        ```bash
        compact-memory compress --file input.txt --engine <shared_engine_id> --budget <value>
        ```
    *   **Python API:**
        ```python
        from compact_memory import get_compression_engine

        engine_instance = get_compression_engine("<shared_engine_id>")()
        compressed_mem, trace = engine_instance.compress(text, budget)
        # ...
        ```

By following these guidelines, you can contribute to and benefit from a growing ecosystem of powerful context compression tools for LLMs.
