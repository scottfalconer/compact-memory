from __future__ import annotations

import os
from pathlib import Path


def _disk_usage(path: Path) -> int:
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


def _install_models(
    embed_model: str = "all-MiniLM-L6-v2", chat_model: str = "distilgpt2"
) -> str:
    """Download the default embedding and chat models."""
    try:
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForCausalLM, AutoTokenizer

        SentenceTransformer(embed_model)
        AutoTokenizer.from_pretrained(chat_model)
        AutoModelForCausalLM.from_pretrained(chat_model)
    except Exception as exc:  # pragma: no cover - network / file errors
        return f"error installing models: {exc}"

    return f"installed {embed_model} and {chat_model}"


def _path_suggestions(prefix: str, limit: int = 5) -> list[str]:
    """Return file path completions for ``prefix``."""
    path = Path(prefix).expanduser()
    base = path.parent if path.name else path
    pattern = path.name + "*"
    try:
        items = [p for p in base.glob(pattern)]
    except OSError:
        items = []
    suggestions = []
    for item in items:
        suggestion = str(item)
        if item.is_dir():
            suggestion += "/"
        suggestions.append(suggestion)
        if len(suggestions) >= limit:
            break
    return suggestions


def _brain_path_suggestions(base: Path, prefix: str, limit: int = 5) -> list[str]:
    """Return sub directories under ``base`` that look like brains."""
    suggestions: list[str] = []
    for p in base.glob(prefix + "*"):
        if (p / "meta.yaml").exists():
            suggestions.append(str(p))
            if len(suggestions) >= limit:
                break
    return suggestions


__all__ = [
    "_disk_usage",
    "_install_models",
    "_path_suggestions",
    "_brain_path_suggestions",
]
