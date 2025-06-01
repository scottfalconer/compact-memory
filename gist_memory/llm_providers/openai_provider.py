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
    }

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
        api_key = llm_kwargs.pop("api_key", None)
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            openai.api_key = api_key
        messages = [{"role": "user", "content": prompt}]
        resp = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            **llm_kwargs,
        )
        return resp.choices[0].message["content"].strip()
