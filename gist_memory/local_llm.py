from __future__ import annotations

"""Simple local LLM wrapper for chat‑style generation."""

from dataclasses import dataclass
import inspect
import logging
import contextlib
from typing import Optional, TYPE_CHECKING, Iterable

from .llm_providers_abc import LLMProvider

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from .agent import Agent

from .importance_filter import dynamic_importance_filter
from .token_utils import token_count

try:  # heavy dependency only when needed
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - optional
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore


@dataclass
class LocalChatModel(LLMProvider):
    """Wrap a local `transformers` causal LM for offline chat."""

    model_name: str = "tiny-gpt2"
    max_new_tokens: int = 100

    tokenizer: Optional["AutoTokenizer"] = None  # set in ``__post_init__``
    model: Optional["AutoModelForCausalLM"] = None
    _loaded: bool = False

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
        """Ensure heavy dependencies are available but defer loading."""
        global AutoModelForCausalLM, AutoTokenizer
        if AutoModelForCausalLM is None or AutoTokenizer is None:
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

    # ------------------------------------------------------------------
    def load_model(self) -> None:
        """Load the tokenizer and model if not already loaded."""
        if self._loaded:
            return
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            self.__post_init__()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, local_files_only=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, local_files_only=True
            )
        except Exception as exc:  # pragma: no cover – depends on local files
            raise RuntimeError(
                f"Error: Local Chat Model '{self.model_name}' not found. "
                "Please run: gist-memory download-chat-model "
                f"--model-name {self.model_name} to install it."
            ) from exc
        self._loaded = True

    def unload_model(self) -> None:
        """Unload the underlying tokenizer and model to free memory."""
        self.tokenizer = None
        self.model = None
        self._loaded = False

    @contextlib.contextmanager
    def loaded(self):
        """Context manager that loads and unloads the model."""
        self.load_model()
        try:
            yield self
        finally:
            self.unload_model()

    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if not self._loaded or self.tokenizer is None or self.model is None:
            self.load_model()

    # ------------------------------------------------------------------
    # LLMProvider API
    # ------------------------------------------------------------------
    def get_token_budget(self, model_name: str, **kwargs) -> int:
        if model_name != self.model_name:
            other = LocalChatModel(
                model_name=model_name, max_new_tokens=self.max_new_tokens
            )
            with other.loaded():
                return other._context_length()
        with self.loaded():
            return self._context_length()

    def count_tokens(self, text: str, model_name: str, **kwargs) -> int:
        if model_name != self.model_name:
            other = LocalChatModel(
                model_name=model_name, max_new_tokens=self.max_new_tokens
            )
            with other.loaded():
                return token_count(other.tokenizer, text)  # type: ignore[arg-type]
        with self.loaded():
            return token_count(self.tokenizer, text)

    def generate_response(
        self,
        prompt: str,
        model_name: str,
        max_new_tokens: int,
        **llm_kwargs,
    ) -> str:
        if model_name != self.model_name:
            other = LocalChatModel(model_name=model_name, max_new_tokens=max_new_tokens)
            with other.loaded():
                return other.reply(prompt)
        old = self.max_new_tokens
        self.max_new_tokens = max_new_tokens
        try:
            return self.reply(prompt)
        finally:
            self.max_new_tokens = old

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reply(self, prompt: str) -> str:
        """Generate a reply to ``prompt`` trimming context if necessary."""
        self._ensure_loaded()

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

        if token_count(self.tokenizer, prompt) > max_input_len:
            excess = len(ids) - max_input_len
            old_ids, keep_ids = ids[:excess], ids[excess:]
            old_text = self.tokenizer.decode(old_ids, skip_special_tokens=True)
            keep_text = self.tokenizer.decode(keep_ids, skip_special_tokens=True)
            filtered = dynamic_importance_filter(old_text)
            prompt = (filtered + "\n" + keep_text).strip()

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
        """Truncate prompt with a recap if it exceeds context length."""

        self._ensure_loaded()

        max_len = self._context_length()
        input_tokens = token_count(self.tokenizer, prompt)
        logging.debug("[prepare_prompt] input tokens=%d max=%d", input_tokens, max_len)
        tokens = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        if input_tokens <= max_len:
            logging.debug("[prepare_prompt] no truncation needed")
            return prompt

        old_tokens = tokens[:-recent_tokens]
        recent_tokens_ids = tokens[-recent_tokens:]
        old_text = self.tokenizer.decode(old_tokens, skip_special_tokens=True)
        recent_text = self.tokenizer.decode(recent_tokens_ids, skip_special_tokens=True)
        logging.debug(
            "[prepare_prompt] old=%d recent=%d",
            len(old_tokens),
            len(recent_tokens_ids),
        )

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

        output = recap_text + recent_text
        output_tokens = token_count(self.tokenizer, output)
        logging.debug(
            "[prepare_prompt] output tokens=%d (recap=%d recent=%d)",
            output_tokens,
            token_count(self.tokenizer, recap_text),
            token_count(self.tokenizer, recent_text),
        )
        return output


__all__ = ["LocalChatModel"]
