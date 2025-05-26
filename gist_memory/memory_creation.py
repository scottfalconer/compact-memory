class MemoryCreator:
    """Base interface for creating memory text from source text."""

    def create(self, text: str) -> str:
        raise NotImplementedError()


class IdentityMemoryCreator(MemoryCreator):
    """Return the text unchanged."""

    def create(self, text: str) -> str:
        return text
