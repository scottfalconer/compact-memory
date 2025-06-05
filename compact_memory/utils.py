from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List


from .embedding_pipeline import get_embedding_dim
from .vector_store import InMemoryVectorStore


if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from .prototype_engine import PrototypeEngine


def load_memory_container(path: Path) -> "PrototypeEngine":
    """Load a :class:`PrototypeEngine` from ``path``."""

    from .prototype_engine import PrototypeEngine
    from .models import BeliefPrototype, RawMemory
    import json
    import numpy as np

    manifest_file = path / "engine_manifest.json"
    memories_file = path / "memories.json"
    vectors_file = path / "vectors.npy"

    with open(manifest_file, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    meta = manifest.get("meta", {})
    dim = meta.get("embedding_dim", get_embedding_dim())
    normalized = meta.get("normalized", True)
    store = InMemoryVectorStore(embedding_dim=dim, normalized=normalized)
    store.meta = meta
    store.prototypes = [BeliefPrototype(**p) for p in manifest.get("prototypes", [])]
    store.proto_vectors = np.load(vectors_file)

    with open(memories_file, "r", encoding="utf-8") as fh:
        store.memories = [RawMemory(**m) for m in json.load(fh)]

    store.index = {p.prototype_id: i for i, p in enumerate(store.prototypes)}
    store._index_dirty = True

    engine = PrototypeEngine(store)
    return engine


def format_ingest_results(
    agent: "PrototypeEngine", results: Iterable[dict[str, object]]
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
