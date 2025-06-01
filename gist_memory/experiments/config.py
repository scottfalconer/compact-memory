from __future__ import annotations

"""Shared dataclasses for experimentation framework."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence


@dataclass(kw_only=True)
class ExperimentConfig:
    """Generic experiment configuration used across modules."""

    dataset: Path | Sequence[Path] | Callable[[], Any]
    similarity_threshold: float = 0.8
    chunker: Any | None = None
    summary_creator: Any | None = None
    work_dir: Optional[Path] = None
    active_memory_params: Dict[str, Any] | None = None
    compression_strategy: Optional[str] = None
    compression_params: Dict[str, Any] = field(default_factory=dict)
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None
    metrics: List[str] = field(default_factory=list)
    metric_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)


__all__ = ["ExperimentConfig"]
