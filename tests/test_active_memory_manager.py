import pytest
import numpy as np

from compact_memory.strategies.experimental import ActiveMemoryManager, ConversationTurn


def test_history_buffer_does_not_exceed_max_size():
    mgr = ActiveMemoryManager(config_max_history_buffer_turns=3)
    for i in range(5):
        mgr.add_turn(ConversationTurn(text=str(i)))
    assert len(mgr.history) == 3


def test_pruning_retains_turns_with_high_trace_strength():
    mgr = ActiveMemoryManager(
        config_max_history_buffer_turns=3,
        config_pruning_weight_trace_strength=1.0,
        config_pruning_weight_current_activation=0.0,
        config_pruning_weight_recency=0.0,
    )
    t1 = ConversationTurn("t1", trace_strength=0.1)
    t2 = ConversationTurn("t2", trace_strength=0.9)
    t3 = ConversationTurn("t3", trace_strength=0.2)
    t4 = ConversationTurn("t4", trace_strength=0.1)
    for t in (t1, t2, t3, t4):
        mgr.add_turn(t)
    assert t2 in mgr.history
    assert len(mgr.history) == 3


def test_pruning_retains_turns_with_high_current_activation():
    mgr = ActiveMemoryManager(
        config_max_history_buffer_turns=3,
        config_pruning_weight_trace_strength=0.0,
        config_pruning_weight_current_activation=1.0,
        config_pruning_weight_recency=0.0,
    )
    t1 = ConversationTurn("t1", current_activation_level=0.1)
    t2 = ConversationTurn("t2", current_activation_level=0.8)
    t3 = ConversationTurn("t3", current_activation_level=0.2)
    t4 = ConversationTurn("t4", current_activation_level=0.1)
    for t in (t1, t2, t3, t4):
        mgr.add_turn(t)
    assert t2 in mgr.history
    assert len(mgr.history) == 3


def test_pruning_retains_most_recent_turns_if_configured():
    mgr = ActiveMemoryManager(
        config_max_history_buffer_turns=4,
        config_prompt_num_forced_recent_turns=2,
        config_pruning_weight_trace_strength=0.0,
        config_pruning_weight_current_activation=0.0,
        config_pruning_weight_recency=0.0,
    )
    turns = [ConversationTurn(str(i)) for i in range(5)]
    for t in turns:
        mgr.add_turn(t)
    assert turns[-1] in mgr.history
    assert turns[-2] in mgr.history
    assert len(mgr.history) == 4


def test_pruning_correctly_removes_lowest_priority_turn():
    mgr = ActiveMemoryManager(
        config_max_history_buffer_turns=3,
        config_pruning_weight_trace_strength=1.0,
        config_pruning_weight_current_activation=1.0,
        config_pruning_weight_recency=0.0,
    )
    t1 = ConversationTurn("t1", trace_strength=0.2, current_activation_level=0.2)
    t2 = ConversationTurn("t2", trace_strength=0.2, current_activation_level=0.8)
    t3 = ConversationTurn("t3", trace_strength=0.9, current_activation_level=0.2)
    t4 = ConversationTurn("t4", trace_strength=0.1, current_activation_level=0.1)
    for t in (t1, t2, t3, t4):
        mgr.add_turn(t)
    assert t4 not in mgr.history
    assert len(mgr.history) == 3


def test_new_turn_gets_initial_activation_level():
    mgr = ActiveMemoryManager(config_initial_activation=0.9)
    t = ConversationTurn("hi")
    mgr.add_turn(t)
    assert mgr.history[-1].current_activation_level == pytest.approx(0.9)


def test_activation_level_decays_over_subsequent_turns():
    mgr = ActiveMemoryManager(
        config_initial_activation=1.0,
        config_activation_decay_rate=0.1,
    )
    t1 = ConversationTurn("t1")
    mgr.add_turn(t1)
    assert t1.current_activation_level == pytest.approx(1.0)

    t2 = ConversationTurn("t2")
    mgr.add_turn(t2)
    assert t1.current_activation_level == pytest.approx(0.9)

    t3 = ConversationTurn("t3")
    mgr.add_turn(t3)
    assert t1.current_activation_level == pytest.approx(0.81)


def test_relevance_boost_increases_activation_for_semantically_similar_turns():
    mgr = ActiveMemoryManager(
        config_initial_activation=0.0, config_relevance_boost_factor=0.5
    )
    t1 = ConversationTurn("a", turn_embedding=[1.0, 0.0])
    t2 = ConversationTurn("b", turn_embedding=[0.0, 1.0])
    mgr.add_turn(t1)
    mgr.add_turn(t2)
    before = t1.current_activation_level
    mgr.boost_activation_by_relevance(np.array([1.0, 0.0]))
    assert t1.current_activation_level > before
    assert t1.current_activation_level > t2.current_activation_level


def test_relevance_boost_scales_with_similarity_and_boost_factor():
    mgr = ActiveMemoryManager(
        config_initial_activation=0.0, config_relevance_boost_factor=2.0
    )
    emb1 = np.array([1.0, 0.0])
    emb2 = np.array([1.0, 1.0]) / np.sqrt(2)
    t1 = ConversationTurn("a", turn_embedding=emb1.tolist())
    t2 = ConversationTurn("b", turn_embedding=emb2.tolist())
    mgr.add_turn(t1)
    mgr.add_turn(t2)
    mgr.boost_activation_by_relevance(emb1)
    expected_t1 = 2.0 * 1.0  # similarity 1.0 * factor
    expected_t2 = 2.0 * float(np.dot(emb2, emb1))
    assert t1.current_activation_level == pytest.approx(expected_t1)
    assert t2.current_activation_level == pytest.approx(expected_t2)


def test_activation_does_not_fall_below_floor_if_set():
    mgr = ActiveMemoryManager(
        config_initial_activation=1.0,
        config_activation_decay_rate=0.5,
        config_min_activation_floor=0.2,
    )
    t1 = ConversationTurn("t1")
    t2 = ConversationTurn("t2")
    t3 = ConversationTurn("t3")
    t4 = ConversationTurn("t4")
    mgr.add_turn(t1)
    mgr.add_turn(t2)  # t1 decays to 0.5
    mgr.add_turn(t3)  # t1 decays to 0.25
    mgr.add_turn(t4)  # t1 decays to floor 0.2
    assert t1.current_activation_level >= 0.2


def test_candidate_selection_always_includes_forced_recent_turns():
    mgr = ActiveMemoryManager(
        config_prompt_num_forced_recent_turns=2,
        config_prompt_max_activated_older_turns=0,
    )
    t1 = ConversationTurn("old")
    t2 = ConversationTurn("r1")
    t3 = ConversationTurn("r2")
    for t in (t1, t2, t3):
        mgr.add_turn(t)
    selected = mgr.select_history_candidates_for_prompt(np.array([0.0]))
    assert t2 in selected and t3 in selected
    assert t1 not in selected


def test_candidate_selection_pulls_older_turns_based_on_high_activation_after_boost():
    mgr = ActiveMemoryManager(
        config_prompt_num_forced_recent_turns=1,
        config_prompt_max_activated_older_turns=2,
        config_prompt_activation_threshold_for_inclusion=0.1,
        config_relevance_boost_factor=1.0,
    )
    t1 = ConversationTurn("a", turn_embedding=[1.0, 0.0])
    t2 = ConversationTurn("b", turn_embedding=[0.0, 1.0])
    t3 = ConversationTurn("recent")
    for t in (t1, t2, t3):
        mgr.add_turn(t)
    selected = mgr.select_history_candidates_for_prompt(np.array([1.0, 0.0]))
    assert t1 in selected
    assert t3 in selected
    assert t2 not in selected


def test_candidate_selection_respects_max_activated_older_turns_limit():
    mgr = ActiveMemoryManager(
        config_prompt_max_activated_older_turns=2,
        config_prompt_activation_threshold_for_inclusion=0.0,
    )
    turns = [
        ConversationTurn("t1", current_activation_level=0.9),
        ConversationTurn("t2", current_activation_level=0.8),
        ConversationTurn("t3", current_activation_level=0.7),
    ]
    for t in turns:
        mgr.add_turn(t)
    selected = mgr.select_history_candidates_for_prompt(np.array([0.0]))
    assert len(selected) == 2
    assert turns[0] in selected and turns[1] in selected


def test_candidate_selection_uses_trace_strength_for_tie_breaking_among_activated_turns():
    mgr = ActiveMemoryManager(
        config_prompt_max_activated_older_turns=1,
        config_prompt_activation_threshold_for_inclusion=0.0,
    )
    t1 = ConversationTurn("a", current_activation_level=1.0, trace_strength=0.1)
    t2 = ConversationTurn("b", current_activation_level=1.0, trace_strength=0.9)
    mgr.add_turn(t1)
    mgr.add_turn(t2)
    selected = mgr.select_history_candidates_for_prompt(np.array([0.0]))
    assert selected == [t2]


def test_candidate_selection_filters_by_activation_threshold():
    mgr = ActiveMemoryManager(
        config_prompt_max_activated_older_turns=5,
        config_prompt_activation_threshold_for_inclusion=0.5,
    )
    t1 = ConversationTurn("low", current_activation_level=0.4)
    t2 = ConversationTurn("high", current_activation_level=0.6)
    for t in (t1, t2):
        mgr.add_turn(t)
    selected = mgr.select_history_candidates_for_prompt(np.array([0.0]))
    assert t2 in selected and t1 not in selected
