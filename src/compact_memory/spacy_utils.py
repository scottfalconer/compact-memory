import threading
import re

# Heavy import done lazily in ``get_nlp``

_MODEL_NAME = "en_core_web_sm"
_lock = threading.Lock()
_nlp = None


class _SimpleNLP:
    """Fallback spaCy-like object using regex sentence splitting."""

    pipe_names: list[str] = []

    def __call__(self, text: str):
        sentences = simple_sentences(text)

        class _Doc:
            def __init__(self, sents: list[str]):
                self.sents = [type("Span", (), {"text": s})() for s in sents]

        return _Doc(sentences)


def get_nlp():
    """Load and return the shared spaCy model."""
    global _nlp
    if _nlp is None:
        with _lock:
            if _nlp is None:
                try:
                    import spacy
                except Exception:  # pragma: no cover - spaCy not installed
                    _nlp = _SimpleNLP()
                else:
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
