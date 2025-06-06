import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np

from compact_memory.engines.base import CompressedMemory, CompressionTrace
from compact_memory.engines.ReadAgent.engine import ReadAgentGistEngine
# Assuming SentenceWindowChunker is a reasonable default or for testing
from compact_memory.chunker import SentenceWindowChunker

class TestReadAgentGistEngine(unittest.TestCase):
    def setUp(self):
        # Mock the LLM pipeline
        self.mock_llm_pipeline = MagicMock()

        # Default config for the engine
        self.engine_config = {
            "local_llm_pipeline": self.mock_llm_pipeline,
            "episode_token_limit": 500,
            "gist_length": 50, # Shorter for tests
            "gist_prompt_template": "Summarize: {text} in {gist_length} tokens.",
            "qa_prompt_template": "Context: {context} Question: {question} Answer:",
            "lookup_prompt_template": "Question: {question} Summaries: {summaries} Relevant pages:",
        }
        self.engine = ReadAgentGistEngine(**self.engine_config)
        # For ReadAgent, chunks from chunker are "episodes".
        # We set a simple chunker to make episode definition predictable for these tests.
        # The _paginate_episodes method uses its own logic (split by \n\n) when compress is called directly with a string.
        # The chunker is used when ingest is called.
        self.engine.chunker = SentenceWindowChunker(window_size=1, overlap=0)

    # Test `_paginate_episodes`
    def test_paginate_episodes_double_newline(self):
        text = "Episode 1\n\nEpisode 2\n\nEpisode 3"
        expected = ["Episode 1", "Episode 2", "Episode 3"]
        self.assertEqual(self.engine._paginate_episodes(text), expected)

    def test_paginate_episodes_single_block(self):
        text = "Single episode without double newlines."
        expected = ["Single episode without double newlines."]
        self.assertEqual(self.engine._paginate_episodes(text), expected)

    def test_paginate_episodes_empty_string(self):
        text = ""
        expected = []
        self.assertEqual(self.engine._paginate_episodes(text), expected)

    def test_paginate_episodes_only_newlines(self):
        text = "\n\n\n"
        expected = [] # Should strip and find no content
        self.assertEqual(self.engine._paginate_episodes(text), expected)

    # Test `_generate_gist` (mocking LLM)
    def test_generate_gist_uses_llm(self):
        episode_text = "This is an episode."
        expected_gist = "Simulated gist here."
        self.mock_llm_pipeline.return_value = expected_gist

        gist = self.engine._generate_gist(episode_text)
        self.assertEqual(gist, expected_gist)

        expected_prompt = self.engine_config["gist_prompt_template"].format(
            text=episode_text, gist_length=self.engine_config["gist_length"]
        )
        self.mock_llm_pipeline.assert_called_once_with(expected_prompt)

    def test_generate_gist_empty_episode(self):
        gist = self.engine._generate_gist("")
        self.assertEqual(gist, "")
        self.mock_llm_pipeline.assert_not_called()

    # Test `compress` method - Summarization Path
    def test_compress_summarization_path(self):
        doc_text = "First episode.\n\nSecond episode, which is a bit longer."
        episodes = ["First episode.", "Second episode, which is a bit longer."] # Manually derived based on _paginate_episodes

        self.mock_llm_pipeline.side_effect = [
            "Gist 1",
            "Gist 2 a bit longer"
        ]

        expected_concatenated_gists = "Gist 1\n\n---\n\nGist 2 a bit longer"
        budget = 200

        compressed_memory, trace = self.engine.compress(doc_text, llm_token_budget=budget)

        self.assertEqual(compressed_memory.text, expected_concatenated_gists)
        self.assertFalse(trace.output_summary.get("query_processed", True))
        self.assertEqual(trace.input_summary["original_length"], len(doc_text))

        self.assertEqual(self.mock_llm_pipeline.call_count, 2)
        gist_prompt_1 = self.engine_config["gist_prompt_template"].format(text=episodes[0], gist_length=self.engine_config["gist_length"])
        gist_prompt_2 = self.engine_config["gist_prompt_template"].format(text=episodes[1], gist_length=self.engine_config["gist_length"])

        # Use a loop to check calls because assert_has_calls with side_effect can be tricky
        # if order is not strictly guaranteed by implementation details unrelated to logic.
        calls = [call(gist_prompt_1), call(gist_prompt_2)]
        self.mock_llm_pipeline.assert_has_calls(calls, any_order=True)


    def test_compress_summarization_truncation(self):
        doc_text = "Episode one.\n\nEpisode two."
        # Manually derived based on _paginate_episodes
        episodes = ["Episode one.", "Episode two."]

        self.mock_llm_pipeline.side_effect = ["Short gist 1.", "Short gist 2."]

        budget = 10
        concatenated_gists_untruncated = "Short gist 1.\n\n---\n\nShort gist 2."
        expected_text = concatenated_gists_untruncated[:budget]

        compressed_memory, trace = self.engine.compress(doc_text, llm_token_budget=budget)
        self.assertEqual(compressed_memory.text, expected_text)

        # Check if truncation step was logged
        truncation_logged = False
        for step in trace.steps:
            if step['type'] == 'summary_truncation' or \
               (step['type'] == 'truncation' and 'Concatenated gists' in step['details']): # Older version of trace
                truncation_logged = True
                break
        self.assertTrue(truncation_logged, "Truncation step was not logged in trace.")


    # Test `compress` method - QA Path
    def test_compress_qa_path(self):
        doc_text = "Episode A: Info about apples.\n\nEpisode B: Info about bananas."
        # Manually derived based on _paginate_episodes
        episodes_original = ["Episode A: Info about apples.", "Episode B: Info about bananas."]
        query = "What about apples?"

        self.mock_llm_pipeline.side_effect = [
            "Gist A (apples)",
            "Gist B (bananas)",
            "Page 1",
            "Apples are fruits."
        ]

        budget = 200

        compressed_memory, trace = self.engine.compress(doc_text, llm_token_budget=budget, query=query)

        self.assertEqual(compressed_memory.text, "Apples are fruits.")
        self.assertTrue(trace.output_summary.get("query_processed"))
        self.assertEqual(trace.input_summary["query"], query)

        self.assertEqual(self.mock_llm_pipeline.call_count, 4)

        # Expected prompts
        gist_prompt_A = self.engine_config["gist_prompt_template"].format(text=episodes_original[0], gist_length=self.engine_config["gist_length"])
        gist_prompt_B = self.engine_config["gist_prompt_template"].format(text=episodes_original[1], gist_length=self.engine_config["gist_length"])

        expected_lookup_summaries = "Page 1: Gist A (apples)\nPage 2: Gist B (bananas)"
        lookup_prompt = self.engine_config["lookup_prompt_template"].format(question=query, summaries=expected_lookup_summaries)

        # Context for QA: Original Episode A (selected), Gist B (not selected)
        qa_context = f"{episodes_original[0]}\n\n---\n\nGist B (bananas)"
        qa_prompt = self.engine_config["qa_prompt_template"].format(context=qa_context, question=query)

        # Check that all expected calls were made, order might vary for gists.
        # For QA, the order of (gists -> lookup -> qa_answer) is more fixed.
        actual_calls = self.mock_llm_pipeline.call_args_list

        # Check gist calls
        self.assertIn(call(gist_prompt_A), actual_calls)
        self.assertIn(call(gist_prompt_B), actual_calls)
        # Check lookup call
        self.assertIn(call(lookup_prompt), actual_calls)
        # Check QA call
        self.assertIn(call(qa_prompt), actual_calls)

if __name__ == '__main__':
    unittest.main()
