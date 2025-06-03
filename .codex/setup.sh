#!/usr/bin/env bash
# Setup script executed during Codex container setup.
# Install dependencies used during testing and examples.

set -euo pipefail

# Update package lists
#apt-get update

# Install Python and basic tools
#apt-get install -y --no-install-recommends python3 python3-pip git
#apt-get clean

# Install a minimal subset of dependencies for the test suite. The environment
# already includes Python and common tooling, so we avoid heavy optional
# packages and model downloads during the 300s setup window.
if [ -f requirements.txt ]; then
    pip3 install --prefer-binary \
        openai tiktoken numpy faiss-cpu click>=8.2 tqdm pydantic \
        pyyaml transformers spacy "typer[all]>=0.16.0" portalocker \
        "rich>=13.6"
    # Install the project itself without pulling in extra dependencies.
    pip3 install -e . --no-build-isolation --no-deps
fi

# Tools used by CI for linting and testing
pip3 install flake8 pytest

# CLI dependencies that may not be declared in requirements.txt
# (rich and typer are already installed above)


