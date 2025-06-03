import unittest
from unittest.mock import patch, MagicMock
from typing import List, Any, Dict, Optional # Added Optional

import numpy as np # For embeddings

from gist_memory.active_memory_strategy import ActiveMemoryStrategy
from gist_memory.active_memory_manager import ActiveMemoryManager, ConversationTurn
from gist_memory.compression.strategies_abc import CompressedMemory
from gist_memory.compression.trace import CompressionTrace # CompressionTrace was missing in prompt's example structure

# --- Mocking Utilities ---

def mock_tokenizer_func(text: str) -> List[str]:
    """Simple tokenizer that splits by space."""
    return text.split(' ')

def mock_token_count_func(tokenizer_func: Any, text: str) -> int:
    """Counts tokens using the provided tokenizer_func."""
    if not text: # Handle empty string case for tokenizer
        return 0
    return len(tokenizer_func(text))

mock_embeddings_cache: Dict[str, List[float]] = {}

def get_mock_embedding(text: str, index: int = 0) -> List[float]:
    """
    Creates a simple, somewhat unique embedding based on text length and an index.
    Caches results for consistency.
    """
    if text not in mock_embeddings_cache:
        # Create a simple, somewhat unique embedding vector
        # Using a combination of length, index, and sum of char ordinals for variability
        base_val = float(len(text) + index + sum(ord(c) for c in text[:3])) / 100.0
        mock_embeddings_cache[text] = [base_val, base_val + 0.1, base_val + 0.2, base_val + 0.3, base_val + 0.4]
    return mock_embeddings_cache[text]

def mock_embed_texts_func(texts: List[str]) -> List[Optional[List[float]]]:
    """Mocks the embedding function, returning predictable embeddings."""
    if not texts:
        return []
    # Handle cases where a text might be empty or None if the main code expects that
    return [get_mock_embedding(text, i) if text else None for i, text in enumerate(texts)]


class TestActiveMemoryStrategy(unittest.TestCase):
    def setUp(self):
        """Clear mock embeddings cache before each test."""
        global mock_embeddings_cache
        mock_embeddings_cache = {}

    def test_instantiation_default_params(self):
        """Test strategy instantiation with default parameters."""
        strategy = ActiveMemoryStrategy()
        self.assertIsInstance(strategy.manager, ActiveMemoryManager)
        self.assertEqual(strategy.manager.config_max_history_buffer_turns, 100) # Default AMM value
        self.assertEqual(strategy.id, "active_memory_neuro")

    def test_instantiation_custom_params(self):
        """Test strategy instantiation with custom parameters."""
        strategy = ActiveMemoryStrategy(config_max_history_buffer_turns=5, config_activation_decay_rate=0.5)
        self.assertEqual(strategy.manager.config_max_history_buffer_turns, 5)
        self.assertEqual(strategy.manager.config_activation_decay_rate, 0.5)

    def test_add_single_turn(self):
        """Test adding a single turn to the strategy's manager."""
        strategy = ActiveMemoryStrategy()
        test_text = "Hello, world!"
        test_embedding = get_mock_embedding(test_text)
        strategy.add_turn(test_text, turn_embedding=test_embedding)
        
        self.assertEqual(len(strategy.manager.history), 1)
        added_turn = strategy.manager.history[0]
        self.assertEqual(added_turn.text, test_text)
        self.assertEqual(added_turn.turn_embedding, test_embedding)
        # New turns added via strategy.add_turn will use AMM's config_initial_activation
        self.assertEqual(added_turn.current_activation_level, strategy.manager.config_initial_activation)

    @patch('gist_memory.active_memory_strategy._agent_utils.embed_text', new=mock_embed_texts_func)
    def test_compress_empty_history(self):
        """Test compress with no turns in history."""
        strategy = ActiveMemoryStrategy()
        query_text = "Test query"
        compressed, trace = strategy.compress(query_text, 50, tokenizer=mock_tokenizer_func)
        
        self.assertEqual(compressed.text, "")
        self.assertIsInstance(trace, CompressionTrace)
        self.assertIn(f"Input query for compression context: '{query_text}'", trace.steps)
        self.assertIn("History length before selection: 0", trace.steps)
        self.assertIn("Final turns fitting token budget (50 tokens): 0", trace.steps)

    @patch('gist_memory.active_memory_strategy._agent_utils.embed_text', new=mock_embed_texts_func)
    def test_compress_simple_history_fits_budget(self):
        """Test compress with a few turns that fit within the token budget."""
        strategy = ActiveMemoryStrategy(config_prompt_num_forced_recent_turns=2) # Ensure recent are kept
        
        info1 = "Old important info"
        info2 = "Recent relevant info"
        strategy.add_turn(info1, turn_embedding=get_mock_embedding(info1))
        strategy.add_turn(info2, turn_embedding=get_mock_embedding(info2))
        
        query_text = "Query text"
        # Budget allows both turns (e.g., "Old important info" -> 3 tokens, "Recent relevant info" -> 3 tokens)
        compressed, trace = strategy.compress(query_text, 20, tokenizer=mock_tokenizer_func)
        
        expected_text = f"{info1}\n{info2}" # Order depends on AMM logic (forced recent, then older)
        self.assertEqual(compressed.text, expected_text)
        self.assertLessEqual(mock_token_count_func(mock_tokenizer_func, compressed.text), 20)

    @patch('gist_memory.active_memory_strategy._agent_utils.embed_text', new=mock_embed_texts_func)
    def test_compress_respects_token_budget(self):
        """Test that compressed output respects the llm_token_budget."""
        # No forced recent turns to allow AMM to select based on other factors if budget is tight
        strategy = ActiveMemoryStrategy(config_prompt_num_forced_recent_turns=0)
        
        # Add turns that would exceed the budget if all were included
        strategy.add_turn("This is a long turn one.", turn_embedding=get_mock_embedding("This is a long turn one."))
        strategy.add_turn("This is another long turn two.", turn_embedding=get_mock_embedding("This is another long turn two."))
        strategy.add_turn("Short three.", turn_embedding=get_mock_embedding("Short three."))

        query_text = "Query"
        budget = 5 # Should only fit "Short three." if it's prioritized, or one of the longer ones if not.
                      # AMM's finalize_history_for_prompt processes in order of priority.
                      # With default decay, recency has an impact.
        
        compressed, trace = strategy.compress(query_text, budget, tokenizer=mock_tokenizer_func)
        
        # print(f"Budget test compressed text: '{compressed.text}'") # For debugging if needed
        # print(f"Trace for budget test: {trace.steps}")

        # We expect the most recent, "Short three." (2 tokens), to be included first.
        # If budget was 1, it might be empty if "Short three." couldn't fit after query embedding etc.
        # The mock_token_count_func and tokenizer are very simple.
        # AMM's finalize_history_for_prompt takes turns in order of (forced_recent + older_activated_sorted)
        # and adds them until budget is hit. "Short three." is most recent.
        self.assertIn("Short three.", compressed.text) # Or check specific text based on AMM logic
        self.assertLessEqual(mock_token_count_func(mock_tokenizer_func, compressed.text), budget)


    @patch('gist_memory.active_memory_strategy._agent_utils.embed_text', new=mock_embed_texts_func)
    def test_compress_pruning_max_history(self):
        """Test that history is pruned when config_max_history_buffer_turns is exceeded."""
        strategy = ActiveMemoryStrategy(config_max_history_buffer_turns=2, config_prompt_num_forced_recent_turns=0)
        
        turn1_text = "Turn 1 to be pruned"
        turn2_text = "Turn 2 kept"
        turn3_text = "Turn 3 kept most recent"

        strategy.add_turn(turn1_text, trace_strength=0.1, turn_embedding=get_mock_embedding(turn1_text)) # Low trace to be pruned
        strategy.add_turn(turn2_text, trace_strength=1.0, turn_embedding=get_mock_embedding(turn2_text))
        strategy.add_turn(turn3_text, trace_strength=1.0, turn_embedding=get_mock_embedding(turn3_text))
        
        self.assertEqual(len(strategy.manager.history), 2)
        # Check that the correct turns were kept (Turn 2 and Turn 3 due to pruning logic)
        # AMM prunes based on score (trace_strength, activation, recency).
        # Turn1 has low trace_strength.
        history_texts = [t.text for t in strategy.manager.history]
        self.assertNotIn(turn1_text, history_texts)
        self.assertIn(turn2_text, history_texts)
        self.assertIn(turn3_text, history_texts)

        compressed, _ = strategy.compress("Query", 50, tokenizer=mock_tokenizer_func)
        self.assertNotIn(turn1_text, compressed.text)
        self.assertIn(turn2_text, compressed.text)
        self.assertIn(turn3_text, compressed.text)

    @patch('gist_memory.active_memory_strategy._agent_utils.embed_text', new=mock_embed_texts_func)
    def test_compress_relevance_boosting_surfaces_older_turn(self):
        """Test that relevance boosting brings an older, relevant turn into the prompt."""
        strategy = ActiveMemoryStrategy(
            config_max_history_buffer_turns=3, 
            config_prompt_num_forced_recent_turns=0, # Allow boosting to select older
            config_relevance_boost_factor=2.0, 
            config_activation_decay_rate=0.1, # Some decay
            config_prompt_activation_threshold_for_inclusion=-1.0 # Ensure boosted turns are included
        )

        older_relevant_text = "Information about cats"
        irrelevant_recent1_text = "Talk about dogs"
        irrelevant_recent2_text = "Weather today"

        # Make embeddings distinct for testing
        # get_mock_embedding already makes them distinct based on text and index
        strategy.add_turn(older_relevant_text, trace_strength=1.0, turn_embedding=get_mock_embedding(older_relevant_text, 0))
        strategy.add_turn(irrelevant_recent1_text, trace_strength=0.5, turn_embedding=get_mock_embedding(irrelevant_recent1_text, 1))
        strategy.add_turn(irrelevant_recent2_text, trace_strength=0.5, turn_embedding=get_mock_embedding(irrelevant_recent2_text, 2))
        
        # Query that is semantically similar to the older relevant turn
        query_text = "Tell me about felines" # "felines" is related to "cats"
        # To make this test robust, we need mock_embed_texts_func to produce similar embeddings
        # for query_text and older_relevant_text.
        # Our current get_mock_embedding is based on text length and char codes.
        # We can manually adjust the cache for this specific test case for the query.
        mock_embeddings_cache[query_text] = get_mock_embedding(older_relevant_text, 0) # Make query embedding identical to older_relevant_text

        compressed, trace = strategy.compress(query_text, 50, tokenizer=mock_tokenizer_func)
        
        # print(f"Relevance test compressed: '{compressed.text}'")
        # print(f"Relevance test trace: {trace.steps}")
        # for turn in strategy.manager.history:
        #     print(f"History for relevance: {turn.text}, Activation: {turn.current_activation_level}, Embedding: {turn.turn_embedding}")
        # query_emb = mock_embed_texts_func([query_text])[0]
        # print(f"Query embedding for relevance: {query_emb}")


        self.assertIn(older_relevant_text, compressed.text, "Relevant older turn should be surfaced by boosting.")
        # Depending on token budget and other factors, other turns might also be present or not.
        # The key is that the boosted turn is included.

    def test_save_and_load_learnable_components(self):
        """Test that save/load methods exist and do nothing (as AMM is rule-based)."""
        strategy = ActiveMemoryStrategy()
        try:
            strategy.save_learnable_components("./dummy_path")
            strategy.load_learnable_components("./dummy_path")
        except Exception as e:
            self.fail(f"save/load_learnable_components raised an exception: {e}")


if __name__ == '__main__':
    unittest.main()
