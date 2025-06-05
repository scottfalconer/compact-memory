# Contributing to Compact Memory

First off, thank you for considering contributing to Compact Memory! We welcome contributions from the community to help make this framework better, more robust, and more versatile.

## How Can I Contribute?

There are many ways you can contribute to the project:

*   **Reporting Bugs:** If you find a bug, please open an issue on GitHub, providing as much detail as possible, including steps to reproduce, expected behavior, and actual behavior.
*   **Suggesting Enhancements:** If you have ideas for new features or improvements to existing ones, feel free to open an issue to discuss them.
*   **Code Contributions:**
    *   **Bug Fixes:** Implementing fixes for reported bugs.
    *   **Core Framework Improvements:** Enhancements to the core functionalities of Compact Memory.
    *   **New Compression Engines:** Developing new engines and packaging them as shareable plugins (see "Developing Compression Engines" and "Sharing Engines" in the documentation). We are particularly interested in innovative and effective engines.
    *   **New Validation Metrics:** Adding new metrics for evaluating compression or LLM responses.
*   **Documentation Enhancements:** Improving existing documentation, writing new tutorials, or clarifying API references. Good documentation is crucial for a project's success.
*   **New Examples or Tutorials:** Creating new example scripts or Jupyter notebooks that showcase how to use Compact Memory in different scenarios or integrate it with other tools.

## Getting Started

1.  **Fork the Repository:** Click the "Fork" button on the top right of the GitHub repository page.
2.  **Clone Your Fork:** Clone your forked repository to your local machine:
    ```bash
    git clone https://github.com/YOUR_USERNAME/compact-memory.git
    cd compact-memory
    ```
3.  **Set Up Environment:** We recommend using a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    pip install -e . # Install Compact Memory in editable mode
    ```
    If there are development-specific requirements (e.g., for running tests or linting), install them as well (e.g., `pip install -r requirements-dev.txt` if such a file exists).

4.  **Create a Branch:** Create a new branch for your changes:
    ```bash
    git checkout -b my-feature-branch
    ```
    Choose a descriptive branch name (e.g., `fix-cli-bug`, `add-new-engine-template`).

## Coding Standards

*   **Style:** Please follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code. We use `pre-commit` hooks to help enforce code style (e.g., using Black, Flake8). Before committing, please install and run pre-commit:
    ```bash
    pip install pre-commit
    pre-commit install
    # Now pre-commit will run automatically on every commit.
    # You can also run it manually on all files:
    pre-commit run --all-files
    ```
*   **Docstrings:** Write clear and comprehensive docstrings for all new classes, methods, and functions, following common Python conventions (e.g., Google style or NumPy style).
*   **Type Hinting:** Use type hints for all function signatures and variable annotations where appropriate.

## Testing

*   **Write Tests:** Contributions that add new features or fix bugs should include unit tests or integration tests to verify their correctness.
*   **Run Tests:** Ensure all existing tests pass before submitting your contribution. You can typically run tests using a command like `pytest` from the root directory.
    ```bash
    pytest
    ```
    Refer to the project's documentation for any specific testing setups or commands.

## Pull Request (PR) Process

1.  **Commit Your Changes:** Make your changes and commit them with clear, descriptive commit messages.
    ```bash
    git add .
    git commit -m "feat: Add new cool feature"
    # Or for a fix:
    # git commit -m "fix: Resolve issue #123 by doing X"
    ```
2.  **Push to Your Fork:** Push your changes to your forked repository:
    ```bash
    git push origin my-feature-branch
    ```
3.  **Open a Pull Request:** Go to the original Compact Memory repository on GitHub. You should see a prompt to create a Pull Request from your new branch.
    *   Provide a clear title and description for your PR, explaining the changes you've made and why.
    *   If your PR addresses an existing issue, link to it (e.g., "Closes #123").
4.  **Code Review:** Your PR will be reviewed by the maintainers. Be prepared to discuss your changes and make further modifications if requested.
5.  **Merging:** Once your PR is approved and passes all checks, it will be merged into the main codebase.

## Questions?

If you have any questions about contributing, feel free to open an issue or reach out to the maintainers.

Thank you for your contribution!
