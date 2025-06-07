"""Minimal helpers for testing Compact Memory output with an LLM."""

from __future__ import annotations

from typing import Optional
import os

try:  # optional dependency
    import openai
except Exception:  # pragma: no cover - optional
    openai = None  # type: ignore

try:  # optional dependency
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
except Exception:  # pragma: no cover - optional
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    torch = None  # type: ignore


def run_openai(
    prompt: str,
    model_name: str = "gpt-3.5-turbo",
    *,
    max_new_tokens: int = 150,
    api_key: Optional[str] = None,
) -> str:
    """Return completion from OpenAI chat models.

    Args:
        prompt: The full text prompt to send.
        model_name: Target model identifier.
        max_new_tokens: Maximum tokens to generate.
        api_key: Optional API key. Falls back to the ``OPENAI_API_KEY``
            environment variable.

    Returns:
        The generated completion text.
    """
    if openai is None:
        raise ImportError("openai package is required for run_openai()")
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        openai.api_key = api_key
    messages = [{"role": "user", "content": prompt}]
    resp = openai.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_new_tokens,
    )
    return resp.choices[0].message.content.strip()


def run_local(
    prompt: str, model_name: str = "distilgpt2", *, max_new_tokens: int = 128
) -> str:
    """Return completion from a local transformers model.

    Args:
        prompt: Prompt to send to the model.
        model_name: Hugging Face model identifier.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        The generated completion text.
    """
    if AutoModelForCausalLM is None or AutoTokenizer is None or torch is None:
        raise ImportError(
            "transformers with PyTorch is required for run_local()"
        )  # noqa: E501
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=getattr(tokenizer, "eos_token_id", None),
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if text.startswith(prompt):
        text = text[len(prompt) :]  # noqa: E203
    return text.strip()


def run_llm(
    prompt: str,
    *,
    provider: str = "local",
    model_name: str = "distilgpt2",
    max_new_tokens: int = 128,
    api_key: Optional[str] = None,
) -> str:
    """Dispatch to ``run_openai`` or ``run_local`` based on ``provider``.

    Args:
        prompt: Full prompt text.
        provider: ``"openai"`` or ``"local"``.
        model_name: Model identifier understood by the provider.
        max_new_tokens: Maximum tokens to generate.
        api_key: API key for cloud providers.

    Returns:
        The generated completion text.
    """
    if provider == "openai":
        return run_openai(
            prompt,
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            api_key=api_key,
        )
    return run_local(
        prompt,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
    )


__all__ = ["run_openai", "run_local", "run_llm"]
