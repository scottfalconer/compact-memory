"""Utility functions for model downloads."""

from __future__ import annotations


def download_embedding_model(model_name: str) -> None:
    """Download a SentenceTransformer embedding model."""
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(model_name)


def download_chat_model(model_name: str) -> None:
    """Download a local chat model (tokenizer and weights)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    AutoTokenizer.from_pretrained(model_name)
    AutoModelForCausalLM.from_pretrained(model_name)


__all__ = [
    "download_embedding_model",
    "download_chat_model",
]
