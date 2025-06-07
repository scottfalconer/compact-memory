import unittest
from compact_memory.llm_providers.mock_provider import MockLLMProvider


class TestMockLLMProvider(unittest.TestCase):

    def test_instantiation_default(self):
        """Test basic instantiation with default parameters."""
        provider = MockLLMProvider()
        self.assertIsNotNone(provider)
        self.assertEqual(provider.default_response, "Mocked response")
        self.assertEqual(provider.responses, {})

    def test_instantiation_custom(self):
        """Test instantiation with custom parameters."""
        custom_responses = {"hello": "world"}
        custom_default = "Custom default"
        provider = MockLLMProvider(
            responses=custom_responses, default_response=custom_default
        )
        self.assertEqual(provider.responses, custom_responses)
        self.assertEqual(provider.default_response, custom_default)

    def test_generate_response_predefined(self):
        """Test generate_response for a predefined prompt."""
        provider = MockLLMProvider(responses={"test_prompt": "test_response"})
        response = provider.generate_response(
            "test_prompt", "test_model", max_new_tokens=50
        )
        self.assertEqual(response, "test_response")

    def test_generate_response_default(self):
        """Test generate_response when the prompt is not predefined."""
        provider = MockLLMProvider(default_response="Default answer")
        response = provider.generate_response(
            "unknown_prompt", "test_model", max_new_tokens=50
        )
        self.assertEqual(response, "Default answer")

    def test_add_response(self):
        """Test the add_response method."""
        provider = MockLLMProvider()
        provider.add_response("new_prompt", "new_response")
        self.assertIn("new_prompt", provider.responses)
        self.assertEqual(provider.responses["new_prompt"], "new_response")

        response = provider.generate_response(
            "new_prompt", "test_model", max_new_tokens=50
        )
        self.assertEqual(response, "new_response")

    def test_set_default_response(self):
        """Test the set_default_response method."""
        provider = MockLLMProvider()
        provider.set_default_response("Updated default")
        self.assertEqual(provider.default_response, "Updated default")

        response = provider.generate_response(
            "another_unknown_prompt", "test_model", max_new_tokens=50
        )
        self.assertEqual(response, "Updated default")

    def test_count_tokens_default(self):
        """Test the default count_tokens behavior (split by space)."""
        provider = MockLLMProvider()
        self.assertEqual(provider.count_tokens("hello world", "test_model"), 2)
        self.assertEqual(provider.count_tokens("hello", "test_model"), 1)
        self.assertEqual(
            provider.count_tokens("  hello  world  ", "test_model"), 4
        )  # split behavior
        self.assertEqual(provider.count_tokens("", "test_model"), 0)

    def test_count_tokens_custom(self):
        """Test count_tokens with a custom function (e.g., char count)."""
        custom_counter = lambda text, model_name: len(text)
        provider = MockLLMProvider(token_count_fn=custom_counter)
        self.assertEqual(provider.count_tokens("hello world", "test_model"), 11)
        self.assertEqual(provider.count_tokens("hello", "test_model"), 5)
        self.assertEqual(provider.count_tokens("", "test_model"), 0)

    def test_get_token_budget_default(self):
        """Test the default get_token_budget behavior."""
        provider = MockLLMProvider()
        self.assertEqual(provider.get_token_budget("test_model"), 2048)

    def test_get_token_budget_custom(self):
        """Test get_token_budget with a custom function."""
        custom_budget_fn = lambda model_name: (
            1024 if model_name == "custom_model" else 512
        )
        provider = MockLLMProvider(token_budget_fn=custom_budget_fn)
        self.assertEqual(provider.get_token_budget("custom_model"), 1024)
        self.assertEqual(provider.get_token_budget("other_model"), 512)


if __name__ == "__main__":
    unittest.main()
