"""Configuration dataclasses for the experimentation framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Dict, List

from ..chunker import Chunker
from ..memory_creation import MemoryCreator


DatasetLoader = Callable[[], Iterable[Any]]


@dataclass
class ExperimentConfig:
    """Generic configuration for running experiments."""

    # Either ``dataset`` or ``loader`` should be provided.
    dataset: Optional[Path | str] = None
    loader: Optional[DatasetLoader] = None

    # Compression strategy identifier and its parameters
    compression_strategy: str = "none"
    compression_params: Dict[str, Any] = field(default_factory=dict)

    # Legacy agent ingest options
    similarity_threshold: float = 0.8
    chunker: Optional[Chunker] = None
    summary_creator: Optional[MemoryCreator] = None

    # Details for the language model used during the experiment
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None

    # Validation metrics to compute and their individual configs
    validation_metrics: List[Dict[str, Any]] = field(default_factory=list)

    # Optional experiment specific settings from legacy experiments
    work_dir: Optional[Path] = None
    active_memory_params: Optional[Dict[str, Any]] = None


__all__ = ["ExperimentConfig", "DatasetLoader"]
