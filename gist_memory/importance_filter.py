from __future__ import annotations

"""Utilities for extracting salient lines from transcripts."""

import re
from typing import Iterable, List


class SlotExtractor:
    """Return lines containing WHO/WHAT/WHEN style tags."""

    def __init__(self, tags: Iterable[str] | None = None) -> None:
        self.tags = tuple(tags) if tags else ("WHO:", "WHAT:", "WHEN:")

    def extract_lines(self, text: str) -> List[str]:
        lines = []
        for line in text.splitlines():
            if any(tag in line for tag in self.tags):
                lines.append(line)
        return lines


def dynamic_importance_filter(text: str) -> str:
    """Return only salient lines from ``text``.

    Speaker turns, names, dates and decision phrases are preserved while short
    fillers such as "uh-huh" are dropped. Lines containing WHO/WHAT/WHEN tags
    are extracted via :class:`SlotExtractor`.
    """
    extractor = SlotExtractor()
    salient: List[str] = []
    months = (
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    )
    for line in text.splitlines():
        part = line.strip()
        if not part:
            continue
        if "uh-huh" in part.lower():
            continue
        if any(tag in part for tag in extractor.tags):
            salient.append(line)
            continue
        if ":" in part:
            salient.append(line)
            continue
        if re.search(r"\b(?:" + "|".join(months) + r")\b", part):
            salient.append(line)
            continue
        if re.search(r"\b\d{4}\b", part):
            salient.append(line)
            continue
        if re.search(r"\b[A-Z][a-z]+\b", part):
            salient.append(line)
            continue
        if re.search(r"decision|decided", part, re.IGNORECASE):
            salient.append(line)
            continue
    return "\n".join(salient)


__all__ = ["SlotExtractor", "dynamic_importance_filter"]
