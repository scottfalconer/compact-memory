from __future__ import annotations

from typing import Any, Dict
import os

import openai
import tiktoken

from ..llm_providers_abc import LLMProvider


class OpenAIProvider(LLMProvider):
    """LLMProvider implementation for the OpenAI API."""

    # Basic, easily extensible registry of known model context sizes
    MODEL_TOKEN_LIMITS: Dict[str, int] = {
        "gpt-3.5-turbo": 4096,
        "gpt-4-turbo": 128000,
        "gpt-4.1-nano": 8192,
    }

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self.api_key = api_key
        self.base_url = base_url

    def get_token_budget(self, model_name: str, **kwargs) -> int:
        default = kwargs.get("default", 4096)
        return self.MODEL_TOKEN_LIMITS.get(model_name, default)

    def count_tokens(self, text: str, model_name: str, **kwargs) -> int:
        try:
            enc = tiktoken.encoding_for_model(model_name)
        except Exception:
            enc = tiktoken.get_encoding("gpt2")
        return len(enc.encode(text))

    def generate_response(
        self,
        prompt: str,
        model_name: str,
        max_new_tokens: int,
        **llm_kwargs: Any,
    ) -> str:
        api_key = (
            llm_kwargs.pop("api_key", None)
            or self.api_key
            or os.getenv("OPENAI_API_KEY")
        )
        base_url = (
            llm_kwargs.pop("base_url", None)
            or self.base_url
            or os.getenv("OPENAI_BASE_URL")
        )

        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url

        client = openai.OpenAI(**client_kwargs)

        messages = [{"role": "user", "content": prompt}]
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            **llm_kwargs,
        )
        return resp.choices[0].message.content.strip()
