from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List


from .chunker import SentenceWindowChunker, _CHUNKER_REGISTRY
from .embedding_pipeline import get_embedding_dim, EmbeddingDimensionMismatchError


if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from .memory_container import MemoryContainer


def load_memory_container(path: Path) -> "MemoryContainer":
    raise RuntimeError(
        "Persistent storage support was removed. Provide your own loader."
    )


def format_ingest_results(
    agent: "MemoryContainer", results: Iterable[dict[str, object]]
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


__all__ = ["load_memory_container", "format_ingest_results"]
