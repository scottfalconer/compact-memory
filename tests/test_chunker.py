from compact_memory.chunker import SentenceWindowChunker, AgenticChunker


class DummyTokenizer:
    def encode(self, text):
        return text.split()

    def decode(self, tokens):
        return " ".join(tokens)


def test_sentence_window_chunker_overlap():
    s1 = ("alpha " * 220).strip() + "."
    s2 = ("bravo " * 220).strip() + "."
    s3 = ("charlie " * 220).strip() + "."
    text = f"{s1} {s2} {s3}"
    chunker = SentenceWindowChunker(max_tokens=512, overlap_tokens=32)
    chunker.tokenizer = DummyTokenizer()
    chunks = chunker.chunk(text)
    assert len(chunks) == 2
    tok = chunker.tokenizer.encode(chunks[0])[-32:]
    tok2 = chunker.tokenizer.encode(chunks[1])[:32]
    assert tok == tok2


def test_agentic_chunker_basic():
    text = "A. B. C. D."
    chunker = AgenticChunker(max_tokens=2, sim_threshold=0.1)
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2

