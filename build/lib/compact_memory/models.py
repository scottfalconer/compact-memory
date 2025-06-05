from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, Field, field_validator


class BeliefPrototype(BaseModel):
    """Metadata for a prototype without the vector."""

    prototype_id: str
    vector_row_index: int
    summary_text: str = Field(default="")
    strength: float = 1.0
    confidence: float = 1.0
    creation_ts: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(microsecond=0)
    )
    last_updated_ts: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(microsecond=0)
    )
    constituent_memory_ids: List[str] = Field(default_factory=list)

    # --------------------------------------------------------------
    def _repr_html_(self) -> str:
        """Return a rich HTML representation for Jupyter."""
        rows = """
        <tr><th>Prototype ID</th><td>{pid}</td></tr>
        <tr><th>Strength</th><td>{strength:.2f}</td></tr>
        <tr><th>Confidence</th><td>{conf:.2f}</td></tr>
        <tr><th>Summary</th><td>{summary}</td></tr>
        <tr><th>Memories</th><td>{num}</td></tr>
        <tr><th>Created</th><td>{created}</td></tr>
        <tr><th>Updated</th><td>{updated}</td></tr>
        """.format(
            pid=self.prototype_id,
            strength=self.strength,
            conf=self.confidence,
            summary=self.summary_text,
            num=len(self.constituent_memory_ids),
            created=self.creation_ts.isoformat(),
            updated=self.last_updated_ts.isoformat(),
        )
        return f"<table>{rows}</table>"

    @field_validator("summary_text")
    def _limit_summary(cls, v: str) -> str:
        if len(v) > 256:
            return v[:256]
        return v

    @field_validator("strength")
    def _check_strength(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("strength must be > 0")
        return v

    @field_validator("confidence")
    def _check_confidence(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("confidence must be between 0 and 1")
        return v


class RawMemory(BaseModel):
    """Individual memory record."""

    memory_id: str
    raw_text_hash: str
    assigned_prototype_id: Optional[str] = None
    source_document_id: Optional[str] = None
    creation_ts: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(microsecond=0)
    )
    raw_text: str
    embedding: Optional[List[float]] = None


class ConversationalTurn(BaseModel):
    """Record of a single conversational turn."""

    user_message: str
    agent_response: str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(microsecond=0)
    )
    turn_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    turn_embedding: Optional[List[float]] = None
    trace_strength: float = 1.0
    current_activation_level: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
