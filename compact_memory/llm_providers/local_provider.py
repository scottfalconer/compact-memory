from __future__ import annotations

from typing import Any, Dict

from ..llm_providers_abc import LLMProvider
from ..local_llm import LocalChatModel
from ..token_utils import token_count


class LocalTransformersProvider(LLMProvider):
    """LLMProvider using local Hugging Face models."""

    def __init__(self) -> None:
        self._models: Dict[str, LocalChatModel] = {}

    def _get_model(self, model_name: str, max_new_tokens: int) -> LocalChatModel:
        model = self._models.get(model_name)
        if model is None:
            model = LocalChatModel(
                model_name=model_name,
                max_new_tokens=max_new_tokens,
            )
            self._models[model_name] = model
        elif model.max_new_tokens != max_new_tokens:
            model.max_new_tokens = max_new_tokens
        return model

    def get_token_budget(self, model_name: str, **kwargs) -> int:
        model = self._get_model(
            model_name,
            kwargs.get("max_new_tokens", 100),
        )
        with model.loaded():
            return model._context_length()

    def count_tokens(self, text: str, model_name: str, **kwargs) -> int:
        model = self._get_model(
            model_name,
            kwargs.get("max_new_tokens", 100),
        )
        with model.loaded():
            return token_count(model.tokenizer, text)

    def generate_response(
        self,
        prompt: str,
        model_name: str,
        max_new_tokens: int,
        **llm_kwargs: Any,
    ) -> str:
        model = self._get_model(model_name, max_new_tokens)
        with model.loaded():
            model.max_new_tokens = max_new_tokens
            return model.reply(prompt)
