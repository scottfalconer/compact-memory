from __future__ import annotations

from pathlib import Path


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
    path = Path(prefix)
    if path.is_absolute():
        base = path.parent
        prefix = path.name
    suggestions: list[str] = []
    for p in base.glob(prefix + "*"):
        if (p / "meta.yaml").exists():
            suggestions.append(str(p))
            if len(suggestions) >= limit:
                break
    return suggestions


__all__ = [
    "_install_models",
    "_path_suggestions",
    "_brain_path_suggestions",
]
