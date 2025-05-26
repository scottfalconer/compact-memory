#!/usr/bin/env bash
set -euo pipefail

# Update package lists
apt-get update

# Install Python and pip if not already installed
apt-get install -y --no-install-recommends python3 python3-pip git

# Install Python dependencies if requirements.txt is present
if [ -f requirements.txt ]; then
    pip3 install --no-cache-dir -r requirements.txt
fi

# Pre-download the default local embedding model so it is available offline
python3 - <<'EOF'
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2")
EOF

