from __future__ import annotations

"""Simple local LLM wrapper for chat style generation."""

from dataclasses import dataclass
from typing import Optional

from .importance_filter import dynamic_importance_filter

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
        full = self.tokenizer(prompt, return_tensors="pt")
        ids = full["input_ids"][0]
        if len(ids) > max_len:
            excess = len(ids) - max_len
            old_ids = ids[:excess]
            keep_ids = ids[excess:]
            old_text = self.tokenizer.decode(old_ids, skip_special_tokens=True)
            filtered = dynamic_importance_filter(old_text)
            keep_text = self.tokenizer.decode(keep_ids, skip_special_tokens=True)
            prompt = (filtered + "\n" + keep_text).strip()
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
        )
        prompt_trimmed = self.tokenizer.decode(
            inputs["input_ids"][0], skip_special_tokens=True
        )
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # return only the newly generated portion
        if text.startswith(prompt):
            return text[len(prompt) :].strip()
        if text.startswith(prompt_trimmed):
            return text[len(prompt_trimmed) :].strip()
        return text.strip()

    # ------------------------------------------------------------------
    def prepare_prompt(
        self,
        agent: "Agent",
        prompt: str,
        *,
        recent_tokens: int = 600,
        top_k: int = 3,
    ) -> str:
        """Truncate ``prompt`` with a short recap if it exceeds context length."""

        max_len = getattr(getattr(self.model, "config", None), "n_positions", 1024)
        tokens = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        if len(tokens) <= max_len:
            return prompt

        old_tokens = tokens[:-recent_tokens]
        recent_tokens_ids = tokens[-recent_tokens:]
        old_text = self.tokenizer.decode(old_tokens, skip_special_tokens=True)
        recent_text = self.tokenizer.decode(recent_tokens_ids, skip_special_tokens=True)

        from .memory_creation import DefaultTemplateBuilder
        from .embedding_pipeline import embed_text

        builder = DefaultTemplateBuilder()
        canonical = builder.build(old_text, {})
        vec = embed_text(canonical)
        nearest = agent.store.find_nearest(vec, k=top_k)
        proto_map = {p.prototype_id: p for p in agent.store.prototypes}
        summaries = [
            proto_map[pid].summary_text for pid, _ in nearest if pid in proto_map
        ]

        recap = "; ".join(summaries)
        if recap:
            recap_text = f"<recap> Recent conversation: {recap}\n"
        else:
            recap_text = ""

        return recap_text + recent_text


__all__ = ["LocalChatModel"]
