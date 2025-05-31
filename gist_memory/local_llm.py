from __future__ import annotations

"""Simple local LLM wrapper for chat‑style generation."""

from dataclasses import dataclass
import inspect
import logging
from typing import Optional, TYPE_CHECKING, Iterable, Any

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from .agent import Agent

from .importance_filter import dynamic_importance_filter


class _PlaceholderLoader:
    """Minimal loader so tests can monkeypatch ``from_pretrained``."""

    def from_pretrained(self, *_a, **_k):  # pragma: no cover - sanity fallback
        raise ImportError("transformers is required for LocalChatModel")


AutoModelForCausalLM: Any | None = _PlaceholderLoader()
AutoTokenizer: Any | None = _PlaceholderLoader()


@dataclass
class LocalChatModel:
    """Wrap a local `transformers` causal LM for offline chat."""

    model_name: str = "distilgpt2"
    max_new_tokens: int = 100

    tokenizer: Optional["AutoTokenizer"] = None  # populated in ``__post_init__``
    model: Optional["AutoModelForCausalLM"] = None

    # ------------------------------------------------------------------
    def _context_length(self) -> int:
        """Return context window length for the underlying model."""
        config = getattr(self.model, "config", None)
        attrs: Iterable[str] = (
            "n_positions",
            "n_ctx",
            "max_position_embeddings",
        )
        for name in attrs:
            value = getattr(config, name, None)
            if isinstance(value, int) and value > 0:
                return value
        if self.tokenizer is not None:
            val = getattr(self.tokenizer, "model_max_length", None)
            if isinstance(val, int) and val > 0 and val < 10**6:
                return val
        return 1024

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        global AutoModelForCausalLM, AutoTokenizer
        if (
            isinstance(AutoModelForCausalLM, _PlaceholderLoader)
            and AutoModelForCausalLM.from_pretrained
            is _PlaceholderLoader.from_pretrained
        ) or (
            isinstance(AutoTokenizer, _PlaceholderLoader)
            and AutoTokenizer.from_pretrained is _PlaceholderLoader.from_pretrained
        ):
            try:
                from transformers import (
                    AutoModelForCausalLM as _Model,
                    AutoTokenizer as _Tok,
                )

                AutoModelForCausalLM = _Model
                AutoTokenizer = _Tok
            except Exception as exc:  # pragma: no cover – optional
                raise ImportError(
                    "transformers is required for LocalChatModel"
                ) from exc

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, local_files_only=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, local_files_only=True
            )
            logging.debug("Loaded chat model '%s'", self.model_name)
        except Exception as exc:  # pragma: no cover – depends on local files
            if not isinstance(exc, ImportError):
                raise RuntimeError(
                    f"Error: Local Chat Model '{self.model_name}' not found. "
                    "Please run: gist-memory download-chat-model "
                    f"--model-name {self.model_name} to install it."
                ) from exc
            logging.warning("using dummy chat model: %s", exc)

            class _DummyTok:
                def __call__(
                    self, prompt, return_tensors=None, truncation=None, max_length=None
                ):
                    return {"input_ids": [0]}

                def decode(self, ids, skip_special_tokens=True):
                    return "reply"

            class _DummyModel:
                def generate(self, **kw):
                    return [[0]]

            self.tokenizer = _DummyTok()
            self.model = _DummyModel()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reply(self, prompt: str) -> str:
        """Generate a reply to ``prompt`` trimming context if necessary."""
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("LocalChatModel not initialised")

        # --------------------------------------------------------------
        # Token management
        max_len = self._context_length()
        max_input_len = max_len - self.max_new_tokens

        full = self.tokenizer(prompt, return_tensors="pt")
        ids_raw = full["input_ids"]

        if isinstance(ids_raw, (list, tuple)):
            ids = list(
                ids_raw[0]
                if ids_raw and isinstance(ids_raw[0], (list, tuple))
                else ids_raw
            )
        else:
            try:
                ids = list(ids_raw[0])  # type: ignore[index]
            except Exception:
                ids = [ids_raw]  # type: ignore[list-item]

        if len(ids) > max_input_len:
            excess = len(ids) - max_input_len
            old_ids, keep_ids = ids[:excess], ids[excess:]
            old_text = self.tokenizer.decode(old_ids, skip_special_tokens=True)
            keep_text = self.tokenizer.decode(keep_ids, skip_special_tokens=True)
            filtered = dynamic_importance_filter(old_text)
            prompt = (filtered + "\n" + keep_text).strip()
            logging.debug("Prompt trimmed by %d tokens", excess)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_len,
        )
        prompt_trimmed = self.tokenizer.decode(
            inputs["input_ids"][0], skip_special_tokens=True
        )

        # --------------------------------------------------------------
        # Call generate() robustly across transformers versions
        cls_fn = self.model.__class__.generate
        sig = inspect.signature(cls_fn)

        try:
            if "self" in sig.parameters:
                outputs = cls_fn(
                    self.model,
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=getattr(self.tokenizer, "eos_token_id", None),
                )
            else:
                outputs = cls_fn(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=getattr(self.tokenizer, "eos_token_id", None),
                )
        except TypeError:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=getattr(self.tokenizer, "eos_token_id", None),
            )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.debug("Generated %d tokens", len(outputs[0]))

        # Return only the newly generated portion
        for prefix in (prompt, prompt_trimmed):
            if text.startswith(prefix):
                return text[len(prefix) :].strip()
        return text.strip()

    # ------------------------------------------------------------------
    def prepare_prompt(
        self,
        agent: Agent,
        prompt: str,
        *,
        recent_tokens: int = 600,
        top_k: int = 3,
    ) -> str:
        """Truncate ``prompt`` with a short recap if it exceeds context length."""

        if self.tokenizer is None or self.model is None:
            raise RuntimeError("LocalChatModel not initialised")

        max_len = self._context_length()
        tokens = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        if len(tokens) <= max_len:
            return prompt

        old_tokens = tokens[:-recent_tokens]
        recent_tokens_ids = tokens[-recent_tokens:]
        old_text = self.tokenizer.decode(old_tokens, skip_special_tokens=True)
        recent_text = self.tokenizer.decode(recent_tokens_ids, skip_special_tokens=True)
        logging.debug("Prompt too long (%d tokens), generating recap", len(tokens))

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
        recap_text = f"<recap> Recent conversation: {recap}\n" if recap else ""
        logging.debug("Prompt recap includes %d summaries", len(summaries))

        return recap_text + recent_text


__all__ = ["LocalChatModel"]
