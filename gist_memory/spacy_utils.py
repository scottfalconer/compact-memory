import threading

import spacy

_MODEL_NAME = "en_core_web_sm"
_lock = threading.Lock()
_nlp = None


def get_nlp():
    """Load and return the shared spaCy model."""
    global _nlp
    if _nlp is None:
        with _lock:
            if _nlp is None:
                _nlp = spacy.load(_MODEL_NAME)
    return _nlp
