#!/usr/bin/env bash
set -euo pipefail

# Update package lists
apt-get update

# Install Python and pip if not already installed
apt-get install -y --no-install-recommends python3 python3-pip git

# Install Python dependencies when using the repository
if [ -f requirements.txt ]; then
    pip3 install --no-cache-dir -r requirements.txt
    python3 -m spacy download en_core_web_sm
fi

# Tools for linting and testing
pip3 install --no-cache-dir flake8 pytest

# Ensure CLI dependencies are available even when not listed in requirements
pip3 install --no-cache-dir rich typer portalocker

# Pre-download the default local embedding model so it is available offline
python3 - <<'EOF'
from sentence_transformers import SentenceTransformer

SentenceTransformer("all-MiniLM-L6-v2")
EOF
# Pre-download the default chat model for talk mode
python3 - <<'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer

AutoTokenizer.from_pretrained("distilgpt2")
AutoModelForCausalLM.from_pretrained("distilgpt2")
EOF

