# Compact Memory Agent Guidelines

These instructions guide Codex when modifying this repository.

## Scope
All directories in this repository follow these rules.

### Prototype utilities
Experimental strategies in ``compact_memory/strategies/experimental`` may ship
with their own helper modules. Do not create shared ``prototype_utils`` or
similar packages for multiple experimental strategies. Each strategy should be
self‑contained and, if needed, will eventually live in its own repository.

Tests for these experimental modules should live alongside the code under the
same directory instead of the top‑level ``tests`` package. This keeps each
strategy self‑contained and portable.

## Coding Standards
- Format Python code with **Black** and lint with **Flake8**. Run `pre-commit run --files <files>` before committing.
- Follow **PEP 8** and include type hints for functions and methods.
- Write clear docstrings in the Google style.

## Testing
- Run `pytest` from the repository root after making code changes.
- If tests fail due to missing dependencies in the environment, mention this in the PR summary.

## Commits
- Use descriptive commit messages in the imperative mood (e.g., "fix: handle bad input").
- Group related changes and keep commits focused.

## Pull Requests
- Rebase on the latest `main` before creating a patch.
- Summarize the purpose of the PR and note any limitations encountered during testing.
