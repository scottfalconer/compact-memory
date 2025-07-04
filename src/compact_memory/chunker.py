from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Type, Any, Optional # Added Any, Optional

import nltk

try:  # Compatibility across NLTK versions
    DownloadError = nltk.downloader.DownloadError
except AttributeError:  # pragma: no cover - for newer NLTK
    DownloadError = LookupError
import tiktoken

from .spacy_utils import get_nlp


_CHUNKER_REGISTRY: Dict[str, Type["Chunker"]] = {}


def register_chunker(id: str, cls: Type["Chunker"]) -> None:
    _CHUNKER_REGISTRY[id] = cls


class Chunker(ABC):
    """Interface for chunking text."""

    id: str

    @abstractmethod
    def chunk(self, text: str) -> List[str]: ...  # noqa: E704

    def config(self) -> Dict[str, Any]: # Changed to Dict[str, Any]
        return {}


class SentenceWindowChunker(Chunker):
    """Split text into sentence windows with optional overlap."""

    id = "sentence_window"

    def __init__(self, max_tokens: int = 256, overlap_tokens: int = 32, **kwargs): # Add type hints for kwargs if possible, or leave as Any
        if "window_size" in kwargs:
            max_tokens = kwargs.pop("window_size")
        if "overlap" in kwargs:
            overlap_tokens = kwargs.pop("overlap")
        self.max_tokens: int = max_tokens
        self.overlap_tokens: int = overlap_tokens
        try:
            self.tokenizer: Optional[tiktoken.Encoding] = tiktoken.get_encoding("gpt2")
        except Exception:  # pragma: no cover - offline fallback
            self.tokenizer = None

    def config(self) -> Dict[str, Any]: # Changed to Dict[str, Any]
        return {
            "id": self.id,
            "max_tokens": self.max_tokens,
            "overlap_tokens": self.overlap_tokens,
        }

    def _sentences(self, text: str) -> List[str]:
        doc = get_nlp()(text.strip())
        return [s.text.strip() for s in doc.sents if s.text.strip()]

    def chunk(self, text: str) -> List[str]:
        sents = self._sentences(text)
        if not sents:
            return []
        chunks: List[List[int]] = []
        current: List[int] = []
        for sent in sents:
            if self.tokenizer is not None:
                tokens = self.tokenizer.encode(sent)
            else:
                tokens = sent.split()
            if len(tokens) > self.max_tokens:
                logging.warning("long sentence split")
                for i in range(0, len(tokens), self.max_tokens):
                    sub = tokens[i : i + self.max_tokens]  # noqa: E203
                    if current:
                        chunks.append(current)
                        current = []
                    chunks.append(sub)
                continue
            if len(current) + len(tokens) > self.max_tokens:
                chunks.append(current)
                overlap = current[-self.overlap_tokens :] if self.overlap_tokens else []
                current = overlap + tokens
            else:
                current.extend(tokens)
        if current:
            chunks.append(current)
        if self.tokenizer is not None:
            return [self.tokenizer.decode(c) for c in chunks]
        else:
            # c is List[int] if tokenizer was None and tokens were List[int] from sent.split()
            # However, if tokenizer is None, tokens = sent.split() makes tokens List[str].
            # The error occurs if self.tokenizer is None, and tokens are List[int] which is not the case.
            # Let's re-check the logic for when self.tokenizer is None.
            # tokens = sent.split() makes List[str]. So c is List[str].
            # The error `Argument 1 to "join" of "str" has incompatible type "list[int]"; expected "Iterable[str]"`
            # implies that `c` CAN be `List[int]`.
            # This happens if `tokens = self.tokenizer.encode(sent)` was used, but then `self.tokenizer` would not be `None`.
            # The current code:
            # if self.tokenizer is not None:
            #     tokens = self.tokenizer.encode(sent) -> List[int]
            # else:
            #     tokens = sent.split() -> List[str]
            # So `current` can be List[int] or List[str]. And `chunks` can be List[List[int]] or List[List[str]].
            # The final return needs to handle this.
            # If chunks contains List[int], then map(str,c) is needed.
            # If chunks contains List[str], then map(str,c) is harmless.
            return [" ".join(map(str, c)) for c in chunks]


class FixedSizeChunker(Chunker):
    """Fallback chunker splitting by characters."""

    id = "fixed_size"

    def __init__(self, size: int = 1024):
        self.size: int = size

    def config(self) -> Dict[str, Any]: # Changed to Dict[str, Any]
        return {"id": self.id, "size": self.size}

    def chunk(self, text: str) -> List[str]:
        return [
            text[i : i + self.size] for i in range(0, len(text), self.size)
        ]  # noqa: E203


class AgenticChunker(Chunker):
    """Segment text into belief-sized ideas using :func:`agentic_split`."""

    id = "agentic"

    def __init__(self, max_tokens: int = 120, sim_threshold: float = 0.3) -> None:
        self.max_tokens: int = max_tokens
        self.sim_threshold: float = sim_threshold

    def config(self) -> Dict[str, Any]: # Changed to Dict[str, Any]
        return {
            "id": self.id,
            "max_tokens": self.max_tokens,
            "sim_threshold": self.sim_threshold, # This makes the value float
        }

    def chunk(self, text: str) -> List[str]:
        from .segmentation import agentic_split

        return agentic_split(
            text, max_tokens=self.max_tokens, sim_threshold=self.sim_threshold
        )


class NltkSentenceChunker(Chunker):
    """Split text into sentences using NLTK and group them into chunks."""

    id = "nltk_sentence"

    def __init__(self, max_tokens: int = 256, **kwargs): # Add type hints for kwargs if possible
        self.max_tokens: int = max_tokens
        try:
            self.tokenizer: Optional[tiktoken.Encoding] = tiktoken.get_encoding("gpt2")
        except Exception:  # pragma: no cover - offline fallback
            self.tokenizer = None

        try:
            nltk.data.find("tokenizers/punkt")
        except DownloadError:  # pragma: no cover - offline fallback
            try:
                nltk.download("punkt", quiet=True)
            except Exception as e:  # pragma: no cover - offline fallback
                logging.warning(
                    f"NLTK punkt download failed: {e}. Falling back to basic sentence splitting."
                )

    def config(self) -> Dict[str, Any]: # Changed to Dict[str, Any]
        return {
            "id": self.id,
            "max_tokens": self.max_tokens,
        }

    def chunk(self, text: str) -> List[str]:
        try:
            sentences = nltk.sent_tokenize(text)
        except Exception:  # pragma: no cover - punkt not available
            # Fallback to basic split if sent_tokenize fails (e.g. punkt not downloaded)
            parts = [s.strip() for s in text.split(".") if s.strip()]
            sentences = [f"{s}." for s in parts]

        if not sentences:
            return []

        chunks: List[str] = []
        current_chunk_sentences: List[str] = []
        current_chunk_tokens: int = 0

        for sent in sentences:
            if self.tokenizer is not None:
                sentence_tokens = self.tokenizer.encode(sent)
                num_sentence_tokens = len(sentence_tokens)
            else:
                # Fallback if tokenizer is not available
                sentence_tokens = sent.split()
                num_sentence_tokens = len(sentence_tokens)

            if num_sentence_tokens > self.max_tokens:
                logging.warning(
                    f"Sentence with {num_sentence_tokens} tokens exceeds max_tokens ({self.max_tokens}). "
                    f"Splitting the sentence."
                )
                # If there's a current chunk, add it
                if current_chunk_sentences:
                    chunks.append(" ".join(current_chunk_sentences))
                    current_chunk_sentences = []
                    current_chunk_tokens = 0

                # Split the long sentence
                if self.tokenizer is not None:
                    for i in range(0, num_sentence_tokens, self.max_tokens):
                        chunks.append(
                            self.tokenizer.decode(
                                sentence_tokens[i : i + self.max_tokens]
                            )
                        )
                else:
                    # Fallback for splitting if tokenizer is not available
                    for i in range(0, num_sentence_tokens, self.max_tokens):
                        chunks.append(
                            " ".join(sentence_tokens[i : i + self.max_tokens])
                        )
                continue

            if current_chunk_tokens + num_sentence_tokens <= self.max_tokens:
                current_chunk_sentences.append(sent)
                current_chunk_tokens += num_sentence_tokens
            else:
                # Current chunk is full, finalize it
                if current_chunk_sentences:
                    chunks.append(" ".join(current_chunk_sentences))

                # Start a new chunk with the current sentence
                current_chunk_sentences = [sent]
                current_chunk_tokens = num_sentence_tokens

        # Add the last remaining chunk
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))

        return chunks


class SemanticChunker(Chunker):
    """Chunk text by paragraphs with token-aware grouping."""

    id = "semantic"

    def __init__(self, max_tokens: int = 512) -> None:
        self.max_tokens: int = max_tokens
        try:
            self.tokenizer: Optional[tiktoken.Encoding] = tiktoken.get_encoding("gpt2")
        except Exception:  # pragma: no cover - offline fallback
            self.tokenizer = None

    def config(self) -> Dict[str, Any]: # Changed to Dict[str, Any]
        return {"id": self.id, "max_tokens": self.max_tokens}

    def _paragraphs(self, text: str) -> List[str]:
        paragraphs: List[str] = []
        current: List[str] = []
        for line in text.splitlines():
            if line.strip():
                current.append(line.strip())
            else:
                if current:
                    paragraphs.append(" ".join(current))
                    current = []
        if current:
            paragraphs.append(" ".join(current))
        return [p for p in paragraphs if p]

    def chunk(self, text: str) -> List[str]:
        paragraphs = self._paragraphs(text)
        if not paragraphs:
            return []
        chunks: List[str] = []
        current: List[str] = []
        current_tokens = 0
        for para in paragraphs:
            tokens = self.tokenizer.encode(para) if self.tokenizer else para.split()
            num_tokens = len(tokens)
            if num_tokens > self.max_tokens:
                if current:
                    chunks.append("\n\n".join(current))
                    current = []
                    current_tokens = 0
                for i in range(0, num_tokens, self.max_tokens):
                    sub = tokens[i : i + self.max_tokens]
                    if self.tokenizer:
                        chunks.append(self.tokenizer.decode(sub))
                    else:
                        chunks.append(" ".join(sub))
                continue
            if current_tokens + num_tokens <= self.max_tokens:
                current.append(para)
                current_tokens += num_tokens
            else:
                if current:
                    chunks.append("\n\n".join(current))
                current = [para]
                current_tokens = num_tokens
        if current:
            chunks.append("\n\n".join(current))
        return chunks


register_chunker(SentenceWindowChunker.id, SentenceWindowChunker)
register_chunker(FixedSizeChunker.id, FixedSizeChunker)
register_chunker(AgenticChunker.id, AgenticChunker)
register_chunker(NltkSentenceChunker.id, NltkSentenceChunker)
register_chunker(SemanticChunker.id, SemanticChunker)


__all__ = [
    "Chunker",
    "SentenceWindowChunker",
    "FixedSizeChunker",
    "AgenticChunker",
    "NltkSentenceChunker",
    "SemanticChunker",
    "register_chunker",
    "_CHUNKER_REGISTRY",
]
