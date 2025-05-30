from __future__ import annotations

"""Simple conflict detection and logging utilities."""

import json
import re
from pathlib import Path

from .models import RawMemory


# naive regex for "X is Y" / "X is not Y" style statements
_PATTERN = re.compile(r"^\s*(.+?)\s+is\s+(not\s+)?(.+?)\.?\s*$", re.IGNORECASE)


def negation_conflict(text_a: str, text_b: str) -> bool:
    """Return ``True`` if ``text_a`` contradicts ``text_b`` by simple negation."""
    ma = _PATTERN.match(text_a)
    mb = _PATTERN.match(text_b)
    if not ma or not mb:
        return False
    subj_a, neg_a, obj_a = ma.group(1).strip().lower(), bool(ma.group(2)), ma.group(3).strip().lower()
    subj_b, neg_b, obj_b = mb.group(1).strip().lower(), bool(mb.group(2)), mb.group(3).strip().lower()
    return subj_a == subj_b and obj_a == obj_b and neg_a != neg_b


class SimpleConflictLogger:
    """Append conflicts to ``conflicts.jsonl`` for HITL review."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, prototype_id: str, mem_a: RawMemory, mem_b: RawMemory) -> None:
        with open(self.path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "prototype_id": prototype_id,
                        "memory_id_a": mem_a.memory_id,
                        "memory_id_b": mem_b.memory_id,
                        "text_a": mem_a.raw_text,
                        "text_b": mem_b.raw_text,
                    }
                )
                + "\n"
            )


__all__ = ["negation_conflict", "SimpleConflictLogger"]
