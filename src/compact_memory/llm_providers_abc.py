from __future__ import annotations

"""Abstract base class for Large Language Model providers."""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Interface for LLM provider implementations."""

    @abstractmethod
    def get_token_budget(self, model_name: str, **kwargs) -> int:
        """Return maximum context window (in tokens) for ``model_name``."""

    @abstractmethod
    def count_tokens(self, text: str, model_name: str, **kwargs) -> int:
        """Return token count for ``text`` using ``model_name`` tokenizer."""

    @abstractmethod
    def generate_response(
        self,
        prompt: str,
        model_name: str,
        max_new_tokens: int,
        **llm_kwargs,
    ) -> str:
        """Generate a text completion from ``prompt``."""
