from __future__ import annotations

"""Detect and log potential memory conflicts."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..models import RawMemory


_NEG_WORDS = {"not", "no", "never", "n't"}


@dataclass
class ConflictRecord:
    prototype_id: str
    memory_a: str
    memory_b: str
    reason: str
    score: float


class ConflictLogger:
    """Append conflict records to ``conflicts.jsonl``."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: ConflictRecord) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(record.__dict__) + "\n")


class ConflictFlagger:
    """Lightweight heuristic conflict detector."""

    def __init__(self, logger: ConflictLogger, sim_threshold: float = 0.7) -> None:
        self.logger = logger
        self.sim_threshold = sim_threshold

    # ------------------------------------------------------------------
    def check_pair(
        self,
        prototype_id: str,
        mem_a: RawMemory,
        vec_a: np.ndarray,
        mem_b: RawMemory,
        vec_b: np.ndarray,
    ) -> None:
        score = float(np.dot(vec_a, vec_b))
        if score < self.sim_threshold:
            return
        reason = self._text_conflict(mem_a.raw_text, mem_b.raw_text)
        if reason:
            rec = ConflictRecord(
                prototype_id=prototype_id,
                memory_a=mem_a.memory_id,
                memory_b=mem_b.memory_id,
                reason=reason,
                score=score,
            )
            self.logger.log(rec)

    # ------------------------------------------------------------------
    def _text_conflict(self, a: str, b: str) -> str | None:
        a_low = a.lower()
        b_low = b.lower()

        nums_a = re.findall(r"\b\d+\b", a_low)
        nums_b = re.findall(r"\b\d+\b", b_low)
        if nums_a and nums_b and nums_a != nums_b:
            return "numeric_mismatch"

        toks_a = a_low.split()
        toks_b = b_low.split()
        neg_a = [t for t in toks_a if t in _NEG_WORDS]
        neg_b = [t for t in toks_b if t in _NEG_WORDS]
        if bool(neg_a) != bool(neg_b):
            clean_a = " ".join(t for t in toks_a if t not in _NEG_WORDS)
            clean_b = " ".join(t for t in toks_b if t not in _NEG_WORDS)
            if clean_a == clean_b:
                return "negation_conflict"

        return None


__all__ = ["ConflictFlagger", "ConflictLogger", "ConflictRecord"]
