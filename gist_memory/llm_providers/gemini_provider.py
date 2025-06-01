from __future__ import annotations

from typing import Any, Dict
import os

import google.generativeai as genai

from ..llm_providers_abc import LLMProvider


class GeminiProvider(LLMProvider):
    """LLMProvider implementation for Google Gemini."""

    MODEL_TOKEN_LIMITS: Dict[str, int] = {
        "gemini-1.5-pro-latest": 1_048_576,
    }

    def get_token_budget(self, model_name: str, **kwargs) -> int:
        default = kwargs.get("default", 8192)
        return self.MODEL_TOKEN_LIMITS.get(model_name, default)

    def count_tokens(self, text: str, model_name: str, **kwargs) -> int:
        model = genai.GenerativeModel(model_name)
        return model.count_tokens(text).total_tokens

    def generate_response(
        self,
        prompt: str,
        model_name: str,
        max_new_tokens: int,
        **llm_kwargs: Any,
    ) -> str:
        api_key = llm_kwargs.pop("api_key", None)
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(
            prompt, generation_config={"max_output_tokens": max_new_tokens}
        )
        return resp.text.strip()
