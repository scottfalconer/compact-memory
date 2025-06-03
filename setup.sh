#!/usr/bin/env bash
# Convenience script to install dependencies and download models.
# Run this while online to enable offline use of the CLI and demos.
set -euo pipefail

pip install -r requirements.txt
pip install -e . --no-build-isolation
python -m spacy download en_core_web_sm

# Pre-fetch models used in the examples and tests
gist-memory download-model --model-name all-MiniLM-L6-v2
gist-memory download-chat-model --model-name tiny-gpt2
