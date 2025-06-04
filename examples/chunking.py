import re
from typing import Iterable, List, Any # Added Any for langchain_recursive_splitter kwargs

# Copied from compact_memory/spacy_utils.py
def simple_sentences(text: str) -> list[str]:
    pattern = re.compile(r"(?<!\bDr\.)(?<=[.!?])\s+(?=[A-Z])")
    return [s.strip() for s in re.split(pattern, text.strip()) if s.strip()]

# Copied from compact_memory/segmentation.py
def _sentences(text: str) -> List[str]:
    """Split ``text`` into sentences using simple heuristics."""
    sents = simple_sentences(text)

    if len(sents) <= 2:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        merged: List[str] = []
        i = 0
        while i < len(parts):
            part = parts[i]
            if part in {
                "Dr.",
                "Mr.",
                "Mrs.",
                "Ms.",
                "Sr.",
                "Jr.",
                "St.",
                "Prof.",
                "p.m.",
                "a.m.",
            } and i + 1 < len(parts):
                part = part + " " + parts[i + 1]
                i += 1
            merged.append(part.strip())
            i += 1
        sents = [m for m in merged if m]

    if len(sents) <= 1 or ("p.m." in text or "a.m." in text) and len(sents) < 3:
        sents = simple_sentences(text)

    return sents


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def agentic_split(
    text: str, max_tokens: int = 120, sim_threshold: float = 0.3
) -> List[str]:
    """Segment ``text`` into belief-sized chunks using a Jaccard similarity drop."""
    sents = _sentences(text)
    if not sents:
        return []
    chunks: List[List[str]] = []
    current: List[str] = []
    prev_tokens: List[str] = []

    for sent in sents:
        words = sent.split()
        sim = _jaccard(prev_tokens, words)
        too_long = len(current) + len(words) > max_tokens
        if current and (sim < sim_threshold or too_long):
            chunks.append(current)
            current = words
            prev_tokens = words
        else:
            current.extend(words)
            prev_tokens.extend(words)
    if current:
        chunks.append(current)
    return [" ".join(c) for c in chunks]

# New splitter functions
def newline_splitter(text: str) -> List[str]:
    """Splits text by newline characters."""
    return text.split('\n')

def tiktoken_fixed_size_splitter(text: str, chunk_size: int = 512, model_name: str = "gpt-3.5-turbo") -> List[str]:
    """Splits text into fixed-size chunks based on tiktoken token count."""
    try:
        import tiktoken
    except ImportError:
        raise ImportError(
            "tiktoken library not found. Please install it with `pip install tiktoken` "
            "to use the tiktoken_fixed_size_splitter."
        )

    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # If model_name is not found, fallback to a common encoding
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    chunks = []
    start_idx = 0
    while start_idx < len(tokens):
        end_idx = min(start_idx + chunk_size, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]
        chunks.append(encoding.decode(chunk_tokens))
        start_idx = end_idx
    return chunks

def langchain_recursive_splitter(text: str, **kwargs: Any) -> List[str]:
    """Wraps LangChain's RecursiveCharacterTextSplitter.
    kwargs are passed to the RecursiveCharacterTextSplitter constructor.
    Example: langchain_recursive_splitter(text, chunk_size=1000, chunk_overlap=100)
    """
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        raise ImportError(
            "langchain library not found. Please install it with `pip install langchain` "
            "to use the langchain_recursive_splitter."
        )

    splitter = RecursiveCharacterTextSplitter(**kwargs)
    return splitter.split_text(text)

__all__ = [
    "agentic_split",
    "simple_sentences",
    "newline_splitter",
    "tiktoken_fixed_size_splitter",
    "langchain_recursive_splitter"
]
