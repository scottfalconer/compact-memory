from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List

from .vector_store import InMemoryVectorStore
from .chunker import SentenceWindowChunker, _CHUNKER_REGISTRY
from .embedding_pipeline import get_embedding_dim


def get_disk_usage(path: Path) -> int:
    """Return total size of files under ``path`` in bytes."""
    size = 0
    for root, _, files in os.walk(path):
        for name in files:
            fp = Path(root) / name
            try:
                size += fp.stat().st_size
            except OSError:
                pass
    return size


if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from .agent import Agent


def load_agent(path: Path) -> Agent:
    """Return an :class:`Agent` loaded from ``path``.

    If the stored embedding dimension does not match the current model,
    the store is re-initialized with the correct dimension.
    """
    from .agent import Agent

    dim = get_embedding_dim()
    store = InMemoryVectorStore(embedding_dim=dim, path=str(path))

    chunker_id = store.meta.get("chunker", "sentence_window")
    chunker_cls = _CHUNKER_REGISTRY.get(chunker_id, SentenceWindowChunker)
    tau = float(store.meta.get("tau", 0.8))
    return Agent(store, chunker=chunker_cls(), similarity_threshold=tau)


def format_ingest_results(
    agent: "Agent", results: Iterable[dict[str, object]]
) -> List[str]:
    """Return user-friendly messages for ``agent.add_memory`` results."""

    proto_map = {p.prototype_id: p for p in agent.store.prototypes}
    lines: List[str] = []
    for res in results:
        if res.get("duplicate"):
            lines.append("Duplicate text skipped")
            continue
        action = "Spawned new prototype" if res.get("spawned") else "Updated prototype"
        sim = res.get("sim")
        proto = proto_map.get(res.get("prototype_id", ""))
        summary = proto.summary_text if proto else ""
        sim_str = f" (similarity: {sim:.2f})" if sim is not None else ""
        lines.append(f"{action} {res['prototype_id']}{sim_str}. Summary: '{summary}'")
    return lines


__all__ = ["load_agent", "get_disk_usage", "format_ingest_results"]
