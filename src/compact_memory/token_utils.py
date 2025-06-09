from __future__ import annotations

"""Helper utilities for tokenizer interaction."""

from typing import Any, List


class SimpleTokenizer:
    """Fallback tokenizer that splits text on whitespace."""

    def encode(self, text: str) -> List[str]:
        """Return a list of tokens from ``text``."""
        return text.split()

    def decode(self, tokens: List[Any]) -> str:
        """Join ``tokens`` into a string."""
        return " ".join(str(t) for t in tokens)


def tokenize_text(tokenizer: Any, text: str) -> List[int]:
    """Return a flat list of token ids for ``text`` using ``tokenizer``."""
    if hasattr(tokenizer, "tokenize"):
        try:
            tokens = tokenizer.tokenize(text)
        except Exception:
            try:
                tokens = tokenizer(text, return_tensors=None).get("input_ids", [])
            except Exception:
                tokens = tokenizer(text)
    elif hasattr(tokenizer, "encode"):
        tokens = tokenizer.encode(text)
    else:
        try:
            tokens = tokenizer(text, return_tensors=None).get("input_ids", [])
        except Exception:
            tokens = tokenizer(text)

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


def split_by_tokens(tokenizer: Any, text: str, max_tokens: int) -> List[str]:
    """Split ``text`` into chunks of at most ``max_tokens`` tokens."""
    tokens = tokenize_text(tokenizer, text)
    if len(tokens) <= max_tokens:
        return [text]

    parts: List[str] = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i : i + max_tokens]
        if hasattr(tokenizer, "decode"):
            try:
                parts.append(tokenizer.decode(chunk, skip_special_tokens=True))
                continue
            except Exception:
                pass
        if all(isinstance(t, str) for t in chunk):
            parts.append(" ".join(chunk))
        else:
            parts.append(" ".join(text.split()[i : i + max_tokens]))
    return parts


__all__ = [
    "SimpleTokenizer",
    "tokenize_text",
    "token_count",
    "truncate_text",
    "split_by_tokens",
]
