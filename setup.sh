#!/usr/bin/env bash
# Convenience script to install dependencies and optional example models.
set -euo pipefail

pip install --prefer-binary \
    openai tiktoken numpy faiss-cpu click>=8.2 tqdm pydantic \
    pyyaml transformers spacy "typer[all]>=0.16.0" portalocker \
    "rich>=13.6"

# Optional heavy dependencies required for the full test suite
if [[ "${FULL_INSTALL:-0}" == "1" ]]; then
    pip install torch sentence-transformers google-generativeai evaluate
fi
# Install project without automatically pulling optional heavy deps
pip install -e . --no-build-isolation --no-deps
