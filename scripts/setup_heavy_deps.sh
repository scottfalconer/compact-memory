#!/usr/bin/env bash
# Install heavy dependencies and pre-download models for example metrics.
set -euo pipefail
pip install -e ".[embedding,local]" --no-build-isolation
python - <<'PY'
from compact_memory.model_utils import download_embedding_model, download_chat_model
models = ["all-MiniLM-L6-v2", "multi-qa-mpnet-base-dot-v1"]
for m in models:
    try:
        print(f"Downloading {m}...")
        download_embedding_model(m)
    except Exception as exc:
        print(f"Warning: could not download {m}: {exc}")
try:
    print("Downloading gpt-4.1-nano chat model...")
    download_chat_model("gpt-4.1-nano")
except Exception as exc:
    print(f"Warning: could not download chat model: {exc}")
PY

