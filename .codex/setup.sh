#!/usr/bin/env bash
# Setup script executed during Codex container setup.
# Install dependencies used during testing and examples.

set -euo pipefail

# Update package lists
#apt-get update

# Install Python and basic tools
#apt-get install -y --no-install-recommends python3 python3-pip git
#apt-get clean

# Install Python dependencies for the repository and download spaCy model
if [ -f requirements.txt ]; then
    pip3 install -r requirements.txt
    # Install the package in editable mode so the CLI is available for demos
    pip3 install -e . --no-build-isolation
    # Download the spaCy model needed by the chunkers
    python3 -m spacy download en_core_web_sm
fi

# Tools used by CI for linting and testing
pip3 install flake8 pytest

# CLI dependencies that may not be declared in requirements.txt
pip3 install rich typer portalocker


