from typing import Iterable, List

from .spacy_utils import get_nlp
import re
from .spacy_utils import get_nlp, simple_sentences


def _sentences(text: str) -> List[str]:
    """Split text into sentences using spaCy."""
    doc = get_nlp()(text.strip())
    sents = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    if len(sents) <= 2:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        merged: List[str] = []
        i = 0
        while i < len(parts):
            part = parts[i]
            if part in {"Dr.", "Mr.", "Mrs.", "Ms.", "Sr.", "Jr.", "St.", "Prof.", "p.m.", "a.m."} and i + 1 < len(parts):
                part = part + " " + parts[i + 1]
                i += 1
            merged.append(part.strip())
            i += 1
        sents = [m for m in merged if m]
    nlp = get_nlp()
    try:
        doc = nlp(text.strip())
        sents = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    except Exception:  # pragma: no cover - fallback
        sents = []
    if "parser" not in nlp.pipe_names:
        sents = simple_sentences(text)
    if len(sents) <= 1 or ("p.m." in text or "a.m." in text) and len(sents) < 3:
        sents = simple_sentences(text)
    return sents


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
