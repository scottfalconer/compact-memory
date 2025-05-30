import pytest

from gist_memory.active_memory_manager import ActiveMemoryManager, ConversationTurn


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
