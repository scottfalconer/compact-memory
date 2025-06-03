import threading
import re

# Heavy import done lazily in ``get_nlp``

_MODEL_NAME = "en_core_web_sm"
_lock = threading.Lock()
_nlp = None


def get_nlp():
    """Load and return the shared spaCy model."""
    global _nlp
    if _nlp is None:
        with _lock:
            if _nlp is None:
                import spacy
                try:
                    _nlp = spacy.load(_MODEL_NAME)
                except Exception:  # pragma: no cover - fallback path
                    nlp = spacy.blank("en")
                    nlp.add_pipe("sentencizer")
                    _nlp = nlp
    return _nlp


def simple_sentences(text: str) -> list[str]:
    pattern = re.compile(r"(?<!\bDr\.)(?<=[.!?])\s+(?=[A-Z])")
    return [s.strip() for s in re.split(pattern, text.strip()) if s.strip()]
