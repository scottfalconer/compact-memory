# Compact Memory Agent Guidelines

These instructions guide Codex when modifying this repository.

This project is currently pre-release. Backward compatibility is not
a concern and APIs may change freely.

## Scope
All directories in this repository follow these rules.

### Prototype utilities
Experimental engines may include helper modules alongside the code that
implements them. Keep each engine self‑contained — avoid shared
``prototype_utils`` or other common packages across experimental engines.
If needed, these engines will eventually live in their own repositories.

Tests for these experimental modules should reside next to the code rather than
in the top‑level ``tests`` package. This keeps each engine self‑contained and
portable.

## Coding Standards
- Format Python code with **Black** and lint with **Flake8**. Run `pre-commit run --files <files>` before committing.
- Follow **PEP 8** and include type hints for functions and methods.
- Write clear docstrings in the Google style.
- Name compression engine `id` strings using lowercase `snake_case` so they can be used with the `--engine` CLI option.

## Testing
- Run `pytest` from the repository root after making code changes.
- If tests fail due to missing dependencies in the environment, mention this in the PR summary.

## Commits
- Use descriptive commit messages in the imperative mood (e.g., "fix: handle bad input").
- Group related changes and keep commits focused.

## Pull Requests
- Rebase on the latest `main` before creating a patch.
- Summarize the purpose of the PR and note any limitations encountered during testing.

## LLM Integration
The `local_llm.py` module and `llm_providers` package are essential for Colab examples. Do not remove them when cleaning up the codebase.
