"""Lightweight chunking interface."""

from typing import Callable, List

ChunkFn = Callable[[str], List[str]]

__all__ = ["ChunkFn"]
