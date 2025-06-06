import unittest
from unittest.mock import patch, ANY
from typer.testing import CliRunner
from compact_memory.cli.main import app
from compact_memory.config import DEFAULT_CONFIG
# from compact_memory.llm_models_config import LLM_MODELS_CONFIG_PATH # Not strictly needed if we assume config exists

from compact_memory.llm_providers import MockLLMProvider # Added
from compact_memory.engines.ReadAgent.engine import ReadAgentGistEngine
from compact_memory.chunker import SentenceWindowChunker
from compact_memory.engines.base import CompressedMemory

class TestReadAgentGistEngine(unittest.TestCase):
    def setUp(self):
        self.mock_llm_provider = MockLLMProvider()

        # Default config for the engine
        # Using consistent prompt templates for predictable mock setup
        self.gist_length = 30 # Using a smaller, consistent gist_length for tests
        self.gist_prompt_template = f"Summarize this text in about {self.gist_length} tokens: {{text}}"
        self.qa_prompt_template = "Context: {context} Question: {question} Answer:"
        self.lookup_prompt_template = "Question: {question} Summaries: {summaries} Relevant pages:"

        self.engine_config = {
            "llm_provider": self.mock_llm_provider, # Changed from local_llm_pipeline
            "episode_token_limit": 500,
            "gist_length": self.gist_length,
            "gist_prompt_template": self.gist_prompt_template,
            "qa_prompt_template": self.qa_prompt_template,
            "lookup_prompt_template": self.lookup_prompt_template,
            # Add model names and max tokens as used in ReadAgentGistEngine __init__
            "gist_model_name": "test_gist_model",
            "lookup_model_name": "test_lookup_model",
            "qa_model_name": "test_qa_model",
            "lookup_max_tokens": 20,
            "qa_max_new_tokens": 50,
        }
        self.engine = ReadAgentGistEngine(**self.engine_config)
        self.engine.chunker = SentenceWindowChunker(window_size=1, overlap=0) # As in original

    def test_paginate_episodes_double_newline(self):
        text = "Episode 1\n\nEpisode 2\n\nEpisode 3"
        expected = ["Episode 1", "Episode 2", "Episode 3"]
        self.assertEqual(self.engine._paginate_episodes(text), expected)

    def test_generate_gist_with_provider(self):
        episode_text = "This is a test episode for gisting."
        expected_gist = "Mocked gist for the test episode."

        # Construct the exact prompt _generate_gist will create
        prompt = self.gist_prompt_template.format(text=episode_text)
        self.mock_llm_provider.add_response(prompt, expected_gist)

        gist = self.engine._generate_gist(episode_text)
        self.assertEqual(gist, expected_gist)

    def test_generate_gist_with_provider_empty_episode(self):
        # _generate_gist should return "" for empty text before trying to call provider
        gist = self.engine._generate_gist("")
        self.assertEqual(gist, "")
        # Ensure no calls were made to the provider for an empty episode
        # This requires checking if specific prompts were called, which is hard without knowing the prompt.
        # Instead, we rely on the fact that add_response wasn't called for an empty-text-prompt.
        # A more robust way would be to check call counts on a spy if MockLLMProvider supported it.

    def test_generate_gist_simulation_if_no_provider(self):
        engine_no_provider = ReadAgentGistEngine(
            llm_provider=None,
            gist_length=self.gist_length,
            gist_prompt_template=self.gist_prompt_template
        )
        episode_text = "Test episode for simulation."
        expected_simulated_gist = f"Simulated gist for episode: {episode_text[:50]}..."
        gist = engine_no_provider._generate_gist(episode_text)
        self.assertEqual(gist, expected_simulated_gist)

    def test_select_relevant_episodes_with_provider(self):
        question = "What is the capital of France?"
        episode_gists = [(0, "Paris is the capital."), (1, "London is a city.")]

        formatted_summaries = "Page 1: Paris is the capital.\nPage 2: London is a city."
        prompt = self.lookup_prompt_template.format(question=question, summaries=formatted_summaries)
        self.mock_llm_provider.add_response(prompt, "Page 1") # LLM selects Page 1

        selected_indices = self.engine._select_relevant_episodes(question, episode_gists)
        self.assertEqual(selected_indices, [0])

    def test_select_relevant_episodes_simulation_if_no_provider(self):
        engine_no_provider = ReadAgentGistEngine(llm_provider=None)
        question = "Any question"
        episode_gists = [(0, "Gist A"), (1, "Gist B")]
        # Simulation selects the first episode if available
        expected_indices = [0] if episode_gists else []
        selected_indices = engine_no_provider._select_relevant_episodes(question, episode_gists)
        self.assertEqual(selected_indices, expected_indices)

    def test_compress_chunk_with_provider(self):
        chunk_text = "This is a chunk to be gisted."
        expected_gist = "Mocked gist for the chunk."
        prompt = self.gist_prompt_template.format(text=chunk_text)
        self.mock_llm_provider.add_response(prompt, expected_gist)

        gist = self.engine._compress_chunk(chunk_text)
        self.assertEqual(gist, expected_gist)

    def test_compress_summarization_path_with_provider(self):
        doc_text = "First episode summary here.\n\nSecond episode summary follows."
        episodes = self.engine._paginate_episodes(doc_text) # ["First episode summary here.", "Second episode summary follows."]

        # Setup mock gists for each episode
        gist1_text = "Gist for episode 1"
        prompt1 = self.gist_prompt_template.format(text=episodes[0])
        self.mock_llm_provider.add_response(prompt1, gist1_text)

        gist2_text = "Gist for episode 2"
        prompt2 = self.gist_prompt_template.format(text=episodes[1])
        self.mock_llm_provider.add_response(prompt2, gist2_text)

        expected_concatenated_gists = f"{gist1_text}\n\n---\n\n{gist2_text}"
        budget = 200 # Assume this budget is enough

        compressed_memory, trace = self.engine.compress(doc_text, llm_token_budget=budget)
        self.assertEqual(compressed_memory.text, expected_concatenated_gists)
        self.assertFalse(trace.output_summary.get("query_processed", True))

    def test_compress_summarization_path_simulation_no_provider(self):
        engine_no_provider = ReadAgentGistEngine(
            llm_provider=None,
            gist_length=self.gist_length,
            gist_prompt_template=self.gist_prompt_template
        )
        doc_text = "Sim episode 1.\n\nSim episode 2."
        episodes = engine_no_provider._paginate_episodes(doc_text)

        sim_gist1 = f"Simulated gist for episode: {episodes[0][:50]}..."
        sim_gist2 = f"Simulated gist for episode: {episodes[1][:50]}..."
        expected_text = f"{sim_gist1}\n\n---\n\n{sim_gist2}"
        budget = 200

        compressed_memory, _ = engine_no_provider.compress(doc_text, llm_token_budget=budget)
        self.assertEqual(compressed_memory.text, expected_text)


    def test_compress_qa_path_with_provider(self):
        doc_text = "Episode Alpha: Apples are red.\n\nEpisode Beta: Bananas are yellow."
        query = "Tell me about apples."
        budget = 100

        episodes = self.engine._paginate_episodes(doc_text)
        # episodes = ["Episode Alpha: Apples are red.", "Episode Beta: Bananas are yellow."]

        # 1. Mock Gist Generation
        gist_alpha_text = "Info: Apples"
        prompt_gist_alpha = self.gist_prompt_template.format(text=episodes[0])
        self.mock_llm_provider.add_response(prompt_gist_alpha, gist_alpha_text)

        gist_beta_text = "Info: Bananas"
        prompt_gist_beta = self.gist_prompt_template.format(text=episodes[1])
        self.mock_llm_provider.add_response(prompt_gist_beta, gist_beta_text)

        # 2. Mock Episode Selection
        # Formatted summaries for the lookup prompt
        formatted_summaries = f"Page 1: {gist_alpha_text}\nPage 2: {gist_beta_text}"
        prompt_lookup = self.lookup_prompt_template.format(question=query, summaries=formatted_summaries)
        # Assume LLM selects the first page (Alpha) as relevant
        self.mock_llm_provider.add_response(prompt_lookup, "Page 1")

        # 3. Mock QA Answer
        # Context will be: Full text of selected Episode Alpha, Gist of non-selected Episode Beta
        qa_context = f"{episodes[0]}\n\n---\n\n{gist_beta_text}"
        prompt_qa = self.qa_prompt_template.format(context=qa_context, question=query)
        expected_qa_answer = "Apples are indeed red and quite tasty."
        self.mock_llm_provider.add_response(prompt_qa, expected_qa_answer)

        compressed_memory, trace = self.engine.compress(doc_text, llm_token_budget=budget, query=query)

        self.assertEqual(compressed_memory.text, expected_qa_answer)
        self.assertTrue(trace.output_summary.get("query_processed"))

    def test_compress_qa_path_simulation_no_provider(self):
        engine_no_provider = ReadAgentGistEngine(
            llm_provider=None,
            gist_length=self.gist_length,
            gist_prompt_template=self.gist_prompt_template,
            qa_prompt_template=self.qa_prompt_template,
            lookup_prompt_template=self.lookup_prompt_template
        )
        doc_text = "Sim QA ep1. Content here.\n\nSim QA ep2. More content."
        query = "What is in ep1?"
        budget = 100

        # Simulation will:
        # 1. Create simulated gists
        # 2. Select the first episode (index 0) by default in _select_relevant_episodes simulation
        # 3. Construct context using full text of ep1 and simulated gist of ep2
        # 4. Generate a simulated QA answer
        expected_simulated_answer = f"Simulated answer for query: {query[:50]}..."

        compressed_memory, _ = engine_no_provider.compress(doc_text, llm_token_budget=budget, query=query)
        self.assertEqual(compressed_memory.text, expected_simulated_answer)

    def test_compress_empty_input_text(self):
        compressed_memory, trace = self.engine.compress("", llm_token_budget=100)
        self.assertEqual(compressed_memory.text, "")
        self.assertEqual(trace.output_summary["compressed_length"], 0)
        self.assertEqual(trace.input_summary["num_episodes"], 0)

    def test_compress_summarization_truncation_with_provider(self):
        doc_text = "Very long episode one for summary.\n\nVery long episode two for summary."
        episodes = self.engine._paginate_episodes(doc_text)

        gist1 = "This is the first gist and it is quite long."
        gist2 = "This is the second gist, also very lengthy."

        prompt1 = self.gist_prompt_template.format(text=episodes[0])
        self.mock_llm_provider.add_response(prompt1, gist1)
        prompt2 = self.gist_prompt_template.format(text=episodes[1])
        self.mock_llm_provider.add_response(prompt2, gist2)

        concatenated_gists = f"{gist1}\n\n---\n\n{gist2}"

        budget = 20 # A small budget to force truncation
        expected_text = concatenated_gists[:budget]

        compressed_memory, trace = self.engine.compress(doc_text, llm_token_budget=budget)
        self.assertEqual(compressed_memory.text, expected_text)

        truncation_logged = any(step['type'] == 'summary_truncation' for step in trace.steps)
        self.assertTrue(truncation_logged, "Summary truncation was not logged.")

    @patch("compact_memory.llm_providers.local_provider.LocalTransformersProvider.generate_response")
    def test_readagent_gist_engine_cli_uses_default_llm(self, mock_generate_response):
        runner = CliRunner()
        # This is the text that the ReadAgentGistEngine will output as the compressed result,
        # as it's the return value of the mocked LLM call.
        mock_llm_output = "Mocked LLM response for tiny-gpt2"
        mock_generate_response.return_value = mock_llm_output

        original_model_id = DEFAULT_CONFIG["default_model_id"]
        DEFAULT_CONFIG["default_model_id"] = "tiny-gpt2"
        # Assumption: 'tiny-gpt2' is in llm_models_config.yaml and configured to use the 'local' provider.
        # The 'local' provider is LocalTransformersProvider.
        # ReadAgentGistEngine's default gist_model_name is 'distilgpt2' and default gist_length is 100.

        try:
            result = runner.invoke(
                app,
                [
                    "compress",
                    "--text",
                    "Test document for ReadAgentGistEngine CLI.",
                    "--budget", # CLI budget is for the final output. ReadAgentGistEngine produces one LLM output as its result.
                    "50",     # If LLM output is >50, it would be truncated by the CLI wrapper normally.
                              # Here, mock_llm_output is shorter than 50.
                    "--engine",
                    "readagent_gist",
                    # No --model-id, so it should pick up default_model_id from global config to select the provider.
                ],
            )

            self.assertEqual(result.exit_code, 0, msg=f"CLI Error: {result.stdout} {result.exception}")
            # The output of the compress command should be the LLM's response.
            self.assertIn(mock_llm_output, result.stdout)

            # Check that the mocked LocalTransformersProvider's generate_response was called correctly.
            # ReadAgentGistEngine, when not configured via its own params (which CLI doesn't do for one-shot),
            # uses its default gist_model_name ('distilgpt2') and gist_length (100).
            mock_generate_response.assert_any_call(
                prompt=ANY, # The exact prompt depends on the engine's template and input text.
                model_name="distilgpt2",
                max_new_tokens=100
            )

        finally:
            DEFAULT_CONFIG["default_model_id"] = original_model_id

if __name__ == '__main__':
    unittest.main()
