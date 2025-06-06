#!/usr/bin/env bash
# Setup script executed automatically during Codex container setup.
# Not intended for general use; local installations should run setup.sh.
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
# Install the project and its dependencies as declared in pyproject.toml
if [ -f pyproject.toml ]; then
    # Editable install so source changes are picked up without reinstalling
    pip3 install --upgrade setuptools
    pip3 install -e . --no-build-isolation
fi

# Tools used by CI for linting and testing
pip3 install flake8 pytest

# CLI dependencies that may not be declared in requirements.txt
# (rich and typer are already installed above)


