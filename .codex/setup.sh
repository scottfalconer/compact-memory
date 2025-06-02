#!/usr/bin/env bash
# Setup script executed during Codex container setup.
# Codex loses network access after this step, so we download all assets
# needed for tests here to ensure offline execution.

set -euo pipefail

# Update package lists
apt-get update

# Install Python and basic tools
apt-get install -y --no-install-recommends python3 python3-pip git

# Install Python dependencies for the repository and download spaCy model
if [ -f requirements.txt ]; then
    pip3 install --no-cache-dir -r requirements.txt
    # Install the package in editable mode so the CLI is available for demos
    pip3 install -e . --no-build-isolation
    # Download the spaCy model needed by the chunkers
    python3 -m spacy download en_core_web_sm
fi

# Tools used by CI for linting and testing
pip3 install --no-cache-dir flake8 pytest

# CLI dependencies that may not be declared in requirements.txt
pip3 install --no-cache-dir rich typer portalocker

# Pre-download the default local embedding model so tests work offline
python3 - <<'PY'
from sentence_transformers import SentenceTransformer

# Use the fully qualified model name to ensure the cached path
# matches calls within the repo which expect
# "sentence-transformers/all-MiniLM-L6-v2".
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
PY

# Pre-download the default chat model used in talk mode
python3 - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer

AutoTokenizer.from_pretrained("distilgpt2")
AutoModelForCausalLM.from_pretrained("distilgpt2")
PY

# Ensure the GPT-2 tokenizer files are available for tiktoken
python3 - <<'PY'
import tiktoken
tiktoken.get_encoding("gpt2")
PY

