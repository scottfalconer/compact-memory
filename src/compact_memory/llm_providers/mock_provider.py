from typing import Dict, Optional, Callable, Any
from compact_memory.llm_providers_abc import LLMProvider

class MockLLMProvider(LLMProvider):
    """
    A mock LLM provider for testing purposes.
    It allows predefining responses for specific prompts and provides default behaviors.
    """

    def __init__(
        self,
        responses: Optional[Dict[str, str]] = None,
        default_response: str = "Mocked response",
        token_count_fn: Optional[Callable[[str, str], int]] = None,
        token_budget_fn: Optional[Callable[[str], int]] = None,
    ):
        """
        Initializes the MockLLMProvider.

        Args:
            responses: A dictionary mapping prompt strings to predefined responses.
            default_response: The response to return if a prompt is not found in `responses`.
            token_count_fn: An optional function to customize token counting.
                            It should take (text: str, model_name: str) and return int.
            token_budget_fn: An optional function to customize token budget retrieval.
                             It should take (model_name: str) and return int.
        """
        self.responses: Dict[str, str] = responses if responses is not None else {}
        self.default_response: str = default_response

        # Default token count function: simple word count
        self._token_count_fn: Callable[[str, str], int] = token_count_fn if token_count_fn else lambda text, model_name: len(text.split())

        # Default token budget function: fixed value
        self._token_budget_fn: Callable[[str], int] = token_budget_fn if token_budget_fn else lambda model_name: 2048

    def get_token_budget(self, model_name: str, **kwargs: Any) -> int:
        """
        Returns the token budget for the given model.
        Uses the custom `token_budget_fn` if provided, otherwise a default.
        """
        return self._token_budget_fn(model_name)

    def count_tokens(self, text: str, model_name: str, **kwargs: Any) -> int:
        """
        Counts the tokens in the given text for the specified model.
        Uses the custom `token_count_fn` if provided, otherwise a default.
        """
        if not text:
            return 0
        return self._token_count_fn(text, model_name)

    def generate_response(
        self,
        prompt: str,
        model_name: str,
        max_new_tokens: int,
        **llm_kwargs: Any
    ) -> str:
        """
        Generates a response for the given prompt.
        Returns a predefined response if the prompt is in `self.responses`,
        otherwise returns `self.default_response`.

        Args:
            prompt: The input prompt string.
            model_name: The name of the model to use (ignored in this mock).
            max_new_tokens: The maximum number of new tokens to generate (can be used for truncation).
            **llm_kwargs: Additional keyword arguments for the LLM.

        Returns:
            The generated response string.
        """
        response = self.responses.get(prompt, self.default_response)

        # Optional: Truncate based on max_new_tokens (word count for simplicity)
        # This is a simple interpretation; a real tokenizer would be more accurate.
        # For now, we'll just return the full mock response or default.
        # If truncation is desired:
        # response_tokens = response.split()
        # if len(response_tokens) > max_new_tokens:
        #     return " ".join(response_tokens[:max_new_tokens])
        return response

    def add_response(self, prompt: str, response: str) -> None:
        """
        Adds a new predefined response for a specific prompt.
        """
        self.responses[prompt] = response

    def set_default_response(self, response: str) -> None:
        """
        Updates the default response.
        """
        self.default_response = response
