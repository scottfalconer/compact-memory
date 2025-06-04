from typing import Any, Dict, List, Optional

import pytest

from compact_memory.strategies.experimental import (
    ActiveMemoryStrategy,
    ActiveMemoryManager,
)
from compact_memory.compression.trace import CompressionTrace


# --- Mocking Utilities -----------------------------------------------------


def mock_tokenizer_func(text: str) -> List[str]:
    """Simple whitespace tokenizer."""
    return text.split()


def mock_token_count_func(tokenizer_func: Any, text: str) -> int:
    """Count tokens using ``tokenizer_func``."""
    return len(tokenizer_func(text)) if text else 0


mock_embeddings_cache: Dict[str, List[float]] = {}


def get_mock_embedding(text: str, index: int = 0) -> List[float]:
    if text not in mock_embeddings_cache:
        base = float(len(text) + index + sum(ord(c) for c in text[:3])) / 100.0
        mock_embeddings_cache[text] = [base + 0.1 * i for i in range(5)]
    return mock_embeddings_cache[text]


def mock_embed_texts_func(texts: List[str]) -> List[Optional[List[float]]]:
    if not texts:
        return []
    return [get_mock_embedding(t, i) if t else None for i, t in enumerate(texts)]


@pytest.fixture(autouse=True)
def clear_mock_cache() -> None:
    global mock_embeddings_cache
    mock_embeddings_cache = {}
    yield


# --- Tests -----------------------------------------------------------------


def test_instantiation_default_params() -> None:
    strategy = ActiveMemoryStrategy()
    assert isinstance(strategy.manager, ActiveMemoryManager)
    assert strategy.manager.config_max_history_buffer_turns == 100
    assert strategy.id == "active_memory_neuro"


def test_instantiation_custom_params() -> None:
    strategy = ActiveMemoryStrategy(
        config_max_history_buffer_turns=5,
        config_activation_decay_rate=0.5,
    )
    assert strategy.manager.config_max_history_buffer_turns == 5
    assert strategy.manager.config_activation_decay_rate == 0.5


def test_add_single_turn() -> None:
    strategy = ActiveMemoryStrategy()
    text = "Hello, world!"
    embedding = get_mock_embedding(text)
    strategy.add_turn(text, turn_embedding=embedding)

    assert len(strategy.manager.history) == 1
    added = strategy.manager.history[0]
    assert added.text == text
    assert added.turn_embedding == embedding
    assert added.current_activation_level == strategy.manager.config_initial_activation


def test_compress_empty_history(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "compact_memory.strategies.experimental.active_memory_strategy._agent_utils.embed_text",
        mock_embed_texts_func,
    )
    strategy = ActiveMemoryStrategy()
    query = "Test query"
    compressed, trace = strategy.compress(query, 50, tokenizer=mock_tokenizer_func)

    assert compressed.text == ""
    assert isinstance(trace, CompressionTrace)
    assert f"Input query for compression context: '{query}'" in trace.steps
    assert "History length before selection: 0" in trace.steps
    assert "Final turns fitting token budget (50 tokens): 0" in trace.steps


def test_compress_simple_history_fits_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "compact_memory.strategies.experimental.active_memory_strategy._agent_utils.embed_text",
        mock_embed_texts_func,
    )
    strategy = ActiveMemoryStrategy(config_prompt_num_forced_recent_turns=2)
    info1 = "Old important info"
    info2 = "Recent relevant info"
    strategy.add_turn(info1, turn_embedding=get_mock_embedding(info1))
    strategy.add_turn(info2, turn_embedding=get_mock_embedding(info2))

    compressed, _ = strategy.compress("Query text", 20, tokenizer=mock_tokenizer_func)
    assert compressed.text == f"{info1}\n{info2}"
    assert mock_token_count_func(mock_tokenizer_func, compressed.text) <= 20


def test_compress_respects_token_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "compact_memory.strategies.experimental.active_memory_strategy._agent_utils.embed_text",
        mock_embed_texts_func,
    )
    strategy = ActiveMemoryStrategy(config_prompt_num_forced_recent_turns=0)
    strategy.add_turn(
        "This is a long turn one.",
        turn_embedding=get_mock_embedding("This is a long turn one."),
    )
    strategy.add_turn(
        "This is another long turn two.",
        turn_embedding=get_mock_embedding("This is another long turn two."),
    )
    strategy.add_turn("Short three.", turn_embedding=get_mock_embedding("Short three."))

    compressed, _ = strategy.compress("Query", 5, tokenizer=mock_tokenizer_func)
    assert "Short three." in compressed.text
    assert mock_token_count_func(mock_tokenizer_func, compressed.text) <= 5


def test_compress_pruning_max_history(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "compact_memory.strategies.experimental.active_memory_strategy._agent_utils.embed_text",
        mock_embed_texts_func,
    )
    strategy = ActiveMemoryStrategy(
        config_max_history_buffer_turns=2,
        config_prompt_num_forced_recent_turns=0,
    )

    t1 = "Turn 1 to be pruned"
    t2 = "Turn 2 kept"
    t3 = "Turn 3 kept most recent"

    strategy.add_turn(t1, trace_strength=0.1, turn_embedding=get_mock_embedding(t1))
    strategy.add_turn(t2, trace_strength=1.0, turn_embedding=get_mock_embedding(t2))
    strategy.add_turn(t3, trace_strength=1.0, turn_embedding=get_mock_embedding(t3))

    assert len(strategy.manager.history) == 2
    texts = [t.text for t in strategy.manager.history]
    assert t1 not in texts
    assert t2 in texts
    assert t3 in texts

    compressed, _ = strategy.compress("Query", 50, tokenizer=mock_tokenizer_func)
    assert t1 not in compressed.text
    assert t2 in compressed.text
    assert t3 in compressed.text


def test_compress_relevance_boosting_surfaces_older_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "compact_memory.strategies.experimental.active_memory_strategy._agent_utils.embed_text",
        mock_embed_texts_func,
    )
    strategy = ActiveMemoryStrategy(
        config_max_history_buffer_turns=3,
        config_prompt_num_forced_recent_turns=0,
        config_relevance_boost_factor=2.0,
        config_activation_decay_rate=0.1,
        config_prompt_activation_threshold_for_inclusion=-1.0,
    )

    older = "Information about cats"
    ir1 = "Talk about dogs"
    ir2 = "Weather today"

    strategy.add_turn(
        older, trace_strength=1.0, turn_embedding=get_mock_embedding(older, 0)
    )
    strategy.add_turn(
        ir1, trace_strength=0.5, turn_embedding=get_mock_embedding(ir1, 1)
    )
    strategy.add_turn(
        ir2, trace_strength=0.5, turn_embedding=get_mock_embedding(ir2, 2)
    )

    query = "Tell me about felines"
    mock_embeddings_cache[query] = get_mock_embedding(older, 0)

    compressed, _ = strategy.compress(query, 50, tokenizer=mock_tokenizer_func)
    assert older in compressed.text
