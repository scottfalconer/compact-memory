"""Show ActiveMemoryManager selecting turns in a multi-turn dialogue."""

from compact_memory.active_memory_manager import ActiveMemoryManager, ConversationTurn
from compact_memory.embedding_pipeline import MockEncoder
import numpy as np


def main() -> None:
    enc = MockEncoder()
    mgr = ActiveMemoryManager(
        config_prompt_num_forced_recent_turns=1,
        config_prompt_max_activated_older_turns=2,
        config_relevance_boost_factor=1.0,
    )

    history = [
        "I love astronomy and space missions.",
        "Let's discuss the Apollo program.",
        "Do you know who first walked on the moon?",
    ]
    for t in history:
        emb = enc.encode(t)
        mgr.add_turn(ConversationTurn(text=t, turn_embedding=emb.tolist()))

    query = "Tell me more about Apollo 11"
    q_emb = enc.encode(query)
    candidates = mgr.select_history_candidates_for_prompt(q_emb)
    for turn in candidates:
        print(f"Selected: {turn.text} (activation={turn.current_activation_level:.2f})")


if __name__ == "__main__":
    main()
