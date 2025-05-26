class MemoryCreator:
    """Base interface for creating memory text from source text."""

    def create(self, text: str) -> str:
        raise NotImplementedError()


class IdentityMemoryCreator(MemoryCreator):
    """Return the text unchanged."""

    def create(self, text: str) -> str:
        return text


class ExtractiveSummaryCreator(MemoryCreator):
    """Return the first ``max_words`` words of the text.

    This very simple extractor acts as a lightweight summarisation
    strategy that works in offline environments without any additional
    dependencies.  It allows experiments with pluggable memory creation
    engines as mentioned in ``TODO.md``.
    """

    def __init__(self, max_words: int = 50) -> None:
        self.max_words = max_words

    def create(self, text: str) -> str:
        words = text.split()
        return " ".join(words[: self.max_words])

