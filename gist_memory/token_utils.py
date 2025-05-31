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


__all__ = ["tokenize_text", "token_count"]
