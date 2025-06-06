from __future__ import annotations

"""Helper functions for the prototype system strategy."""

import re
from typing import Iterable, Optional


def render_five_w_template(
    text: str,
    *,
    who: Optional[str] = None,
    what: Optional[str] = None,
    when: Optional[str] = None,
    where: Optional[str] = None,
    why: Optional[str] = None,
) -> str:
    """Return ``text`` wrapped in a canonical 5W template.

    Unknown fields are omitted entirely.
    """
    parts: list[str] = []
    if who:
        parts.append(f"WHO: {who};")
    if what:
        parts.append(f"WHAT: {what};")
    if when:
        parts.append(f"WHEN: {when};")
    if where:
        parts.append(f"WHERE: {where};")
    if why:
        parts.append(f"WHY: {why}.")
    parts.append(f"CONTENT: {text}")
    return " ".join(parts)


__all__ = ["render_five_w_template"]
