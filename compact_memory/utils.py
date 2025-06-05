from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List


if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from .prototype_engine import PrototypeEngine


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


__all__ = ["format_ingest_results"]
