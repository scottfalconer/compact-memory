#!/usr/bin/env bash
set -euo pipefail

# Update package lists
apt-get update

# Install Python and pip if not already installed
apt-get install -y --no-install-recommends python3 python3-pip git

# Install Python dependencies if requirements.txt is present
if [ -f requirements.txt ]; then
pip3 install --no-cache-dir -r requirements.txt
python3 -m spacy download en_core_web_sm
else
    pip3 install --no-cache-dir textual
fi

# Ensure CLI dependencies are available even when not listed in requirements
pip3 install --no-cache-dir textual rich typer portalocker

# Pre-download the default local embedding model so it is available offline
python3 -m gist_memory download-model --model-name all-MiniLM-L6-v2
# Pre-download the default chat model for talk mode
python3 -m gist_memory download-chat-model --model-name distilgpt2

