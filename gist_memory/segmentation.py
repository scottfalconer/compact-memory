import re
from typing import List, Iterable


def _sentences(text: str) -> List[str]:
    """Split text into rough sentences."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def agentic_split(text: str, max_tokens: int = 120, sim_threshold: float = 0.3) -> List[str]:
    """Segment ``text`` into belief-sized chunks using a Jaccard similarity drop."""
    sents = _sentences(text)
    if not sents:
        return []
    chunks: List[List[str]] = []
    current: List[str] = []
    prev_tokens: List[str] = []

    for sent in sents:
        words = sent.split()
        sim = _jaccard(prev_tokens, words)
        too_long = len(current) + len(words) > max_tokens
        if current and (sim < sim_threshold or too_long):
            chunks.append(current)
            current = words
            prev_tokens = words
        else:
            current.extend(words)
            prev_tokens.extend(words)
    if current:
        chunks.append(current)
    return [" ".join(c) for c in chunks]


__all__ = ["agentic_split"]
