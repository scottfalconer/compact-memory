"""Shared configuration constants."""

import os

# Default location for on-disk memory store. Can be overridden with the
# ``GIST_MEMORY_PATH`` environment variable.
DEFAULT_MEMORY_PATH = os.environ.get("GIST_MEMORY_PATH", "memory")

__all__ = ["DEFAULT_MEMORY_PATH"]
