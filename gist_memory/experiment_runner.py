from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

from .agent import Agent
from .json_npy_store import JsonNpyVectorStore
from .chunker import Chunker, SentenceWindowChunker
from .memory_creation import MemoryCreator, ExtractiveSummaryCreator
from .embedding_pipeline import embed_text


@dataclass
class ExperimentConfig:
    """Configuration for :func:`run_experiment`."""

    dataset: Path
    similarity_threshold: float = 0.8
    chunker: Optional[Chunker] = None
    summary_creator: Optional[MemoryCreator] = None
    work_dir: Optional[Path] = None


def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Ingest ``config.dataset`` and return metrics."""

    work = config.work_dir or Path(tempfile.mkdtemp())
    dim = int(embed_text(["dim"]).shape[1])
    store = JsonNpyVectorStore(
        path=str(work), embedding_model="experiment", embedding_dim=dim
    )
    agent = Agent(
        store,
        chunker=config.chunker or SentenceWindowChunker(),
        similarity_threshold=config.similarity_threshold,
        summary_creator=config.summary_creator
        or ExtractiveSummaryCreator(max_words=25),
    )

    text = Path(config.dataset).read_text()
    start = time.perf_counter()
    agent.add_memory(text)
    duration = time.perf_counter() - start

    metrics = dict(agent.metrics)
    metrics.update(
        {
            "prototype_count": len(agent.store.prototypes),
            "memory_count": len(agent.store.memories),
            "ingest_seconds": duration,
        }
    )
    agent.store.save()
    return metrics
