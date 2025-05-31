from __future__ import annotations

"""Helper utilities for tokenizer interaction."""

from typing import Any, List


def tokenize_text(tokenizer: Any, text: str) -> List[int]:
    """Return a flat list of token ids for ``text`` using ``tokenizer``."""
    if hasattr(tokenizer, "tokenize"):
        try:
            tokens = tokenizer.tokenize(text)
        except Exception:
            tokens = tokenizer(text, return_tensors=None).get("input_ids", [])
    else:
        tokens = tokenizer(text, return_tensors=None).get("input_ids", [])

    if isinstance(tokens, (list, tuple)):
        if tokens and isinstance(tokens[0], (list, tuple)):
            token_list = list(tokens[0])
        else:
            token_list = list(tokens)
    else:
        token_list = [tokens]
    return token_list


def token_count(tokenizer: Any, text: str) -> int:
    """Return the number of tokens in ``text`` using ``tokenizer``."""
    return len(tokenize_text(tokenizer, text))


def truncate_text(tokenizer: Any, text: str, max_tokens: int) -> str:
    """Return ``text`` truncated to ``max_tokens`` using ``tokenizer``."""
    tokens = tokenize_text(tokenizer, text)
    if len(tokens) <= max_tokens:
        return text
    trimmed = tokens[:max_tokens]
    if hasattr(tokenizer, "decode"):
        try:
            return tokenizer.decode(trimmed, skip_special_tokens=True)
        except Exception:
            pass
    if all(isinstance(t, str) for t in trimmed):
        return " ".join(trimmed)
    return " ".join(text.split()[:max_tokens])


__all__ = ["tokenize_text", "token_count", "truncate_text"]
