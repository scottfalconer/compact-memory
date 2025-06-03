#!/usr/bin/env bash
# Convenience script to install dependencies and optional example models.
set -euo pipefail

pip install -r requirements.txt
pip install -e . --no-build-isolation
python -m spacy download en_core_web_sm
