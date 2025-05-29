from __future__ import annotations

"""Simple local LLM wrapper for chat style generation."""

from dataclasses import dataclass
from typing import Optional

try:  # heavy dependency only when needed
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - optional
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore


@dataclass
class LocalChatModel:
    """Wrap a local `transformers` causal LM for offline chat."""

    model_name: str = "distilgpt2"
    max_new_tokens: int = 100

    def __post_init__(self) -> None:
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ImportError("transformers is required for LocalChatModel")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, local_files_only=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, local_files_only=True
            )
        except Exception as exc:  # pragma: no cover - depends on local files
            raise RuntimeError(
                "Chat model not found. "
                "Run `gist-memory download-chat-model --model-name "
                f"{self.model_name}` to install it"
            ) from exc

    def reply(self, prompt: str) -> str:
        """Generate a reply given ``prompt``.

        If ``prompt`` is longer than the model's maximum context length the
        excess tokens are truncated from the start to avoid generation errors.
        """
        max_len = getattr(getattr(self.model, "config", None), "n_positions", 1024)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
        )
        prompt_trimmed = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # return only the newly generated portion
        if text.startswith(prompt):
            return text[len(prompt) :].strip()
        if text.startswith(prompt_trimmed):
            return text[len(prompt_trimmed) :].strip()
        return text.strip()


__all__ = ["LocalChatModel"]
