from __future__ import annotations

from typing import Any, List, Union, Optional
import re

from compact_memory.token_utils import tokenize_text, truncate_text
from compact_memory.spacy_utils import get_nlp, simple_sentences

from .base import BaseCompressionEngine, CompressedMemory, CompressionTrace
from .registry import register_compression_engine

try:  # pragma: no cover - optional dependency
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
except Exception:  # pragma: no cover - optional dependency
    nltk = None
    nltk_stopwords = None


def _get_stopwords(language: str) -> set[str]:
    """Return a set of stopwords for ``language``."""
    if nltk_stopwords is not None:
        try:
            return set(nltk_stopwords.words(language))
        except Exception:  # pragma: no cover - dataset missing
            if nltk is not None:
                try:
                    nltk.download("stopwords", quiet=True)
                    return set(nltk_stopwords.words(language))
                except Exception:  # pragma: no cover - download failed
                    pass
    # Fallback minimal list
    return {"the", "and", "is", "in", "to", "a", "of", "it", "that", "this"}


_FILLER_WORDS = {"um", "uh", "like", "basically", "actually"}
_TOKEN_RE = re.compile(r"\b\w+\b|[^\w\s]")


class StopwordPrunerEngine(BaseCompressionEngine):
    """Remove stopwords and filler content from text."""

    id = "stopword_pruner"

    def compress(
        self,
        text_or_chunks: Union[str, List[str]],
        llm_token_budget: int | None,
        *,
        tokenizer: Any = None,
        previous_compression_result: Optional[CompressedMemory] = None,
        **kwargs: Any,
    ) -> CompressedMemory:
        text = (
            text_or_chunks
            if isinstance(text_or_chunks, str)
            else " ".join(text_or_chunks)
        )

        cfg = self.config or {}
        preserve_order = cfg.get("preserve_order", True)
        min_len = cfg.get("min_word_length", 1)
        remove_fillers = cfg.get("remove_fillers", True)
        remove_duplicates = cfg.get("remove_duplicates", False)
        language = cfg.get("stopwords_language", "english")

        stop_words = _get_stopwords(language)

        nlp = get_nlp()
        use_spacy = hasattr(nlp, "tokenizer")

        removed_counts = {
            "stopwords": 0,
            "fillers": 0,
            "short": 0,
            "duplicates": 0,
        }

        output_tokens: List[str] = []
        seen_sentences: set[str] = set()
        prev_token_lower: str | None = None

        if use_spacy:
            doc = nlp(text)
            sentences = doc.sents
        else:
            sentences = [
                type("Span", (), {"text": s})() for s in simple_sentences(text)
            ]

        for sent in sentences:
            sent_text = sent.text
            if remove_duplicates and sent_text.lower() in seen_sentences:
                removed_counts["duplicates"] += len(_TOKEN_RE.findall(sent_text))
                continue
            if remove_duplicates:
                seen_sentences.add(sent_text.lower())

            if use_spacy:
                tokens = sent
            else:
                tokens = [
                    type("Tok", (), {"text": t})() for t in _TOKEN_RE.findall(sent_text)
                ]

            for tok in tokens:
                token_text = tok.text
                lower = token_text.lower()

                if getattr(tok, "is_space", False):
                    continue
                if (
                    getattr(tok, "is_punct", False)
                    or _TOKEN_RE.fullmatch(token_text)
                    and not token_text.isalnum()
                ):
                    continue
                # Original stopword and filler logic
                if getattr(tok, "is_stop", False) or lower in stop_words:
                    removed_counts["stopwords"] += 1
                    continue
                if remove_fillers and lower in _FILLER_WORDS:
                    removed_counts["fillers"] += 1
                    continue
                if len(lower) < min_len:
                    removed_counts["short"] += 1
                    continue
                if remove_duplicates and prev_token_lower == lower:
                    removed_counts["duplicates"] += 1
                    continue
                output_tokens.append(token_text)
                prev_token_lower = lower

        if not preserve_order:
            output_tokens = sorted(set(output_tokens))

        compressed_text = " ".join(output_tokens)

        tokenizer = tokenizer or (lambda t: t.split())
        orig_tokens = len(tokenize_text(tokenizer, text))

        if llm_token_budget is not None:
            compressed_text = truncate_text(
                tokenizer, compressed_text, llm_token_budget
            )

        compressed_tokens = len(tokenize_text(tokenizer, compressed_text))

        compressed = CompressedMemory(text=compressed_text)
        trace = CompressionTrace(
            engine_name=self.id,
            strategy_params={"llm_token_budget": llm_token_budget},
            input_summary={"input_length": len(text), "input_tokens": orig_tokens},
            steps=[
                {"type": "remove_stopwords", "removed": removed_counts["stopwords"]},
                {"type": "remove_fillers", "removed": removed_counts["fillers"]},
                {"type": "remove_short", "removed": removed_counts["short"]},
                {"type": "remove_duplicates", "removed": removed_counts["duplicates"]},
            ],
            output_summary={
                "final_length": len(compressed_text),
                "final_tokens": compressed_tokens,
            },
            final_compressed_object_preview=compressed_text[:50],
        )
        compressed.trace = trace
        compressed.engine_id = self.id
        compressed.engine_config = self.config
        return compressed


__all__ = ["StopwordPrunerEngine"]
