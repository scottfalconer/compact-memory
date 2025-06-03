import numpy as np
from datetime import datetime
from compact_memory.models import ConversationalTurn


def test_conversational_turn_creation_with_defaults():
    turn = ConversationalTurn(user_message="hi", agent_response="hello")
    assert turn.user_message == "hi"
    assert turn.agent_response == "hello"
    assert isinstance(turn.timestamp, datetime)
    assert isinstance(turn.trace_strength, float) and turn.trace_strength == 1.0
    assert isinstance(turn.current_activation_level, float)


def test_turn_id_is_unique():
    t1 = ConversationalTurn(user_message="a", agent_response="b")
    t2 = ConversationalTurn(user_message="c", agent_response="d")
    assert t1.turn_id != t2.turn_id
    assert len(t1.turn_id) == 32 and len(t2.turn_id) == 32


def test_turn_embedding_storage_and_retrieval():
    emb = [0.1, 0.2, 0.3]
    turn = ConversationalTurn(user_message="hi", agent_response="hello", turn_embedding=emb)
    assert np.allclose(turn.turn_embedding, emb)
