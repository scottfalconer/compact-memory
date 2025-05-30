from gist_memory.active_memory_manager import (
    ActiveMemoryManager,
    ConversationTurn,
)


class DummyTokenizer:
    def __call__(
        self,
        text,
        return_tensors=None,
        truncation=None,
        max_length=None,
    ):
        return {"input_ids": [text.split()]}


def count_tokens(turns, tok):
    return sum(len(tok(t.text)["input_ids"][0]) for t in turns)


def test_final_prompt_history_text_never_exceeds_token_budget():
    tok = DummyTokenizer()
    mgr = ActiveMemoryManager(config_prompt_num_forced_recent_turns=1)
    turns = [ConversationTurn("a b c"), ConversationTurn("d e f g")]
    result = mgr.finalize_history_for_prompt(turns, 5, tok)
    assert count_tokens(result, tok) <= 5


def test_all_candidates_included_if_they_fit_budget():
    tok = DummyTokenizer()
    mgr = ActiveMemoryManager()
    turns = [ConversationTurn("a b"), ConversationTurn("c d")]
    result = mgr.finalize_history_for_prompt(turns, 10, tok)
    assert result == turns


def test_lower_priority_candidates_are_dropped_when_budget_exceeded():
    tok = DummyTokenizer()
    mgr = ActiveMemoryManager(config_prompt_num_forced_recent_turns=1)
    older = ConversationTurn("a b c")
    recent = ConversationTurn("d e")
    turns = [older, recent]
    result = mgr.finalize_history_for_prompt(turns, 2, tok)
    assert result == [recent]


def test_token_counting_is_accurate_using_llm_tokenizer():
    tok = DummyTokenizer()
    mgr = ActiveMemoryManager()
    turn = ConversationTurn("one two three four")
    result = mgr.finalize_history_for_prompt([turn], 4, tok)
    assert result == [turn]
    result = mgr.finalize_history_for_prompt([turn], 3, tok)
    assert result == []


def test_handles_single_candidate_turn_exceeding_budget():
    tok = DummyTokenizer()
    mgr = ActiveMemoryManager()
    turn = ConversationTurn("one two three four five")
    result = mgr.finalize_history_for_prompt([turn], 4, tok)
    assert result == []
