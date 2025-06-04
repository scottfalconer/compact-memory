import re


def simple_sentences(text: str) -> list[str]:
    pattern = re.compile(r"(?<!\bDr\.)(?<=[.!?])\s+(?=[A-Z])")
    return [s.strip() for s in re.split(pattern, text.strip()) if s.strip()]
