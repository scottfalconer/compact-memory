from __future__ import annotations


class MemoryCreator:
    """Base interface for creating memory text from source text."""

    def create(self, text: str) -> str:
        raise NotImplementedError()

    def create_all(self, text: str) -> list[str]:
        """Return one or more memory texts derived from ``text``."""
        return [self.create(text)]


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


class ChunkMemoryCreator(MemoryCreator):
    """Split text into fixed-size chunks and return them all."""

    def __init__(self, chunk_size: int = 100) -> None:
        self.chunk_size = chunk_size

    def create(self, text: str) -> str:
        return self.create_all(text)[0]

    def create_all(self, text: str) -> list[str]:
        words = text.split()
        chunks = [
            " ".join(words[i : i + self.chunk_size])
            for i in range(0, len(words), self.chunk_size)
        ]
        return chunks


class LLMSummaryCreator(MemoryCreator):
    """Use an OpenAI model to summarise the text."""

    def __init__(self, model: str = "gpt-3.5-turbo") -> None:
        self.model = model

    def create(self, text: str) -> str:
        import openai

        resp = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Summarise the following text."},
                {"role": "user", "content": text},
            ],
        )
        return resp["choices"][0]["message"]["content"].strip()


class AgenticMemoryCreator(MemoryCreator):
    """Segment text into belief-sized ideas using ``agentic_split``."""

    def __init__(self, max_tokens: int = 120, sim_threshold: float = 0.3) -> None:
        self.max_tokens = max_tokens
        self.sim_threshold = sim_threshold

    def create(self, text: str) -> str:
        return self.create_all(text)[0]

    def create_all(self, text: str) -> list[str]:
        from .segmentation import agentic_split

        return agentic_split(
            text, max_tokens=self.max_tokens, sim_threshold=self.sim_threshold
        )


_TEMPLATE_REGISTRY: dict[str, type["TemplateBuilder"]] = {}


def register_template_builder(id: str, cls: type["TemplateBuilder"]) -> None:
    _TEMPLATE_REGISTRY[id] = cls


class TemplateBuilder:
    """Combine extracted slots with a sentence into a canonical string."""

    id: str = "default"

    def build(
        self, sentence: str, slots: dict[str, str]
    ) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def config(self) -> dict[str, int | str]:
        return {"id": self.id}


class DefaultTemplateBuilder(TemplateBuilder):
    """Simple builder concatenating slot values."""

    id = "default"

    def build(self, sentence: str, slots: dict[str, str]) -> str:
        parts = [sentence]
        for key in ("who", "what", "when", "where", "why"):
            val = slots.get(key)
            if val:
                parts.append(f"{key}:{val}")
        return " | ".join(parts)


register_template_builder(DefaultTemplateBuilder.id, DefaultTemplateBuilder)


__all__ = [
    "MemoryCreator",
    "IdentityMemoryCreator",
    "ExtractiveSummaryCreator",
    "ChunkMemoryCreator",
    "LLMSummaryCreator",
    "AgenticMemoryCreator",
    "TemplateBuilder",
    "DefaultTemplateBuilder",
    "register_template_builder",
    "_TEMPLATE_REGISTRY",
]
