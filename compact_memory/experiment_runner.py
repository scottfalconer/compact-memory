from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

from .compression.strategies_abc import CompressionStrategy

from .agent import Agent
from .vector_store import InMemoryVectorStore
from .active_memory_manager import ActiveMemoryManager
from .chunker import Chunker, SentenceWindowChunker
from .memory_creation import MemoryCreator, ExtractiveSummaryCreator
from .experiments.config import ExperimentConfig
from .embedding_pipeline import embed_text, get_embedding_dim


def run_experiment(
    config: ExperimentConfig,
    strategy: CompressionStrategy | None = None,
) -> Dict[str, Any]:
    """Ingest ``config.dataset`` and return metrics."""

    work = config.work_dir or Path(tempfile.mkdtemp())
    dim = get_embedding_dim()
    store = InMemoryVectorStore(embedding_dim=dim, path=str(work))
    if config.active_memory_params:
        store.meta.update(config.active_memory_params)
    params = {k: v for k, v in store.meta.items() if k.startswith("config_")}
    if config.active_memory_params:
        params.update(config.active_memory_params)
    ActiveMemoryManager(**params)
    agent = Agent(
        store,
        chunker=config.chunker or SentenceWindowChunker(),
        similarity_threshold=config.similarity_threshold,
        summary_creator=(
            config.summary_creator or ExtractiveSummaryCreator(max_words=25)
        ),
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


__all__ = ["ExperimentConfig", "run_experiment"]
