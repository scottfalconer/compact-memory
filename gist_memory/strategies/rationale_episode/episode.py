from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List
from uuid import uuid4


@dataclass
class Decision:
    step_id: str
    step_summary: str
    rationale: str
    importance: float = 0.0


@dataclass
class Episode:
    id: str = field(default_factory=lambda: str(uuid4()))
    timeframe: str = ""
    summary_gist: str = ""
    tags: List[str] = field(default_factory=list)
    decisions: List[Decision] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "Episode":
        decisions = [Decision(**d) for d in data.get("decisions", [])]
        return Episode(
            id=data.get("id", str(uuid4())),
            timeframe=data.get("timeframe", ""),
            summary_gist=data.get("summary_gist", ""),
            tags=list(data.get("tags", [])),
            decisions=decisions,
        )
