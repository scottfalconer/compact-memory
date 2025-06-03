# --- Conversational Memory Example using CompactMemoryAgent ---

from compact_memory import (
    CompactMemoryConfig,
    EmbeddingConfig,
    ChunkerConfig,
    MemoryStoreConfig,
    StrategyConfig,
    CompactMemoryAgent,
    LLMProviderAPIConfig # For the agent's response generation
)
from pathlib import Path
import shutil
import time # For unique session IDs
import os # For environment variable

# Assume a lightweight LLM for agent's response generation for this example
# This could be a local model or a mock.
# For simplicity, we'll mock the LLM's responses based on context.

# Use a common default embedding model for tests, allow override via environment variable
DEFAULT_TEST_EMBEDDING_MODEL = os.environ.get("CM_TEST_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def setup_conversational_agent(store_base_path: Path) -> CompactMemoryAgent:
    """Helper function to configure and initialize a CompactMemoryAgent for conversation."""
    store_path = store_base_path / "conversational_store"
    if store_path.exists(): # Clean up for repeatable example
        shutil.rmtree(store_path)

    convo_config = CompactMemoryConfig(
        default_embedding_config=EmbeddingConfig(
            provider="huggingface",
            model_name=DEFAULT_TEST_EMBEDDING_MODEL
        ),
        default_chunker_config=ChunkerConfig(
            type="sentence_window", # Each turn could be a chunk or few chunks
            params={"window_size": 1, "overlap": 0} # Each sentence (turn) is a chunk
        ),
        memory_store_config=MemoryStoreConfig(
            type="default_json_npy",
            path=str(store_path),
            params={}
        ),
        strategies={
            "recall_related_turns": StrategyConfig(
                id="recall_related_turns_instance",
                # Using FirstKStrategy as a simple placeholder.
                # A real conversational agent would benefit from a strategy that retrieves
                # semantically similar past turns or summaries, possibly time-weighted.
                # If the store returns chunks sorted by similarity to the query embedding (of current turn),
                # FirstKStrategy effectively becomes "TopKSimilarityStrategy".
                strategy_class_id="FirstKStrategy",
                params={"k": 3} # Recall up to 3 most relevant/recent chunks (if store sorts by recency/relevance)
            )
        },
        # Configure a default LLM for the agent to generate responses
        default_llm_provider_config=LLMProviderAPIConfig(
            provider="mock_chat_llm", # This will be a simple mock provider (see below)
            model_name="chatty_mcchatface",
            generation_kwargs={"max_new_tokens": 60} # Limit response length from mock
        ),
        default_tokenizer_name="gpt2" # For prompt construction and budgeting by agent
    )

    agent = CompactMemoryAgent(
        config=convo_config,
        default_strategy_id="recall_related_turns" # Default strategy for retrieve_context
    )
    print(f"Conversational Agent initialized. Memory store location: {agent.storage_path}")
    return agent

# Mock LLM Provider for the example
# In a real scenario, this would be an actual LLM provider class imported from
# compact_memory.llm_providers and registered, or handled by get_llm_provider.
class MockChatLLM: # In reality, this would inherit from compact_memory.llm_providers_abc.LLMProvider
    def __init__(self, config: LLMProviderAPIConfig): # config is LLMProviderAPIConfig
        self.config = config
        print(f"MockChatLLM initialized with model: {config.model_name}")

    def generate_response(self, prompt: str, **kwargs) -> str:
        # Simple mock logic: if certain keywords are in prompt, give canned response
        # It tries to simulate using context by checking for keywords from potential context.
        prompt_lower = prompt.lower()
        user_message_part = prompt_lower.split("user message:")[-1]

        if "hello agent" in user_message_part or "hi agent" in user_message_part :
            return "Hello there! How can I help you today?"
        if "weather" in user_message_part:
            if "sunny" in prompt_lower: # Check if context about sunny weather was provided in full prompt
                return "I recall you mentioning it's sunny. That's great! Enjoy the sunshine."
            return "I'm not sure about the weather right now, but I hope it's pleasant where you are!"
        if "python" in user_message_part:
            if "programming" in prompt_lower and "versatile" in prompt_lower: # Check for context
                 return "Python is indeed a versatile programming language, as we've discussed!"
            return "Python is a fascinating topic! What about it interests you?"
        if "remember this" in user_message_part and "meetings are at 3 pm" in user_message_part:
            return "Okay, I'll try to remember that: meetings are at 3 PM."
        if "what was that important thing" in user_message_part:
            if "meetings are at 3 pm" in prompt_lower: # Check if it's in context
                return "You asked me to remember that meetings are at 3 PM."
            return "I don't have a specific important thing you asked me to remember recently in our current context."

        return f"I've processed your message. You said: '{user_message_part.strip()[:50]}...'"

def run_conversation_turn(
    agent: CompactMemoryAgent,
    user_message: str,
    user_id: str,
    session_id: str
    ):
    """Simulates a single turn in a conversation, ingesting and responding."""
    print(f"\n[User: {user_id} | Session: {session_id}] Says: \"{user_message}\"")

    # Process the message:
    # - Ingests the current user_message into memory (ingest_message_flag=True).
    # - Retrieves relevant past context using the agent's default strategy ("recall_related_turns").
    # - Generates a response using the agent's default LLM provider.
    interaction = agent.process_message(
        message=user_message,
        user_id=user_id,
        session_id=session_id,
        generate_response=True,    # Tell the agent to generate a response
        ingest_message_flag=True,  # Ingest this current user message into memory
        # retrieval_strategy_id can be omitted to use agent's default
        # retrieval_budget can be added if the strategy uses it effectively
    )

    if interaction.error_message:
        print(f"[Agent] Error: {interaction.error_message}")
    else:
        # Log the compressed context that was (potentially) used for generation
        # The context is formed from *previous* turns.
        print(f"  [Agent Internal] Context Used (Strategy: {interaction.context_used.strategy_id_used}): "
              f"'{interaction.context_used.compressed_text[:150].strip()}...'")
        print(f"  [Agent] Says: \"{interaction.llm_response}\"")

    return interaction

# --- Example Usage ---
if __name__ == "__main__":
    example_store_base_path = Path("./temp_convo_example_store_docs")

    # This is a workaround for this specific example script to inject the MockChatLLM.
    # In a real application, the 'get_llm_provider' function within the
    # 'compact_memory.llm_providers' module would be responsible for returning
    # an instance of an actual LLM provider class based on the configuration.
    # We are temporarily replacing it for this self-contained example.
    from compact_memory import llm_providers
    original_get_llm_provider = getattr(llm_providers, "get_llm_provider", None)

    # Define a simple factory function that get_llm_provider can be patched with
    def mock_get_llm_provider_for_example(config: LLMProviderAPIConfig):
        if config.provider == "mock_chat_llm":
            return MockChatLLM(config)
        # Optionally, raise an error or return None for other providers in this mock setup
        raise ValueError(f"Mock setup only supports 'mock_chat_llm', not {config.provider}")

    llm_providers.get_llm_provider = mock_get_llm_provider_for_example

    try:
        # 1. Setup the conversational agent
        convo_agent = setup_conversational_agent(example_store_base_path)

        # 2. Simulate a conversation
        user_id = "user_convo_1"
        session_id = f"session_{int(time.time())}" # Create a unique session ID

        print("\n--- Starting Conversation ---")
        run_conversation_turn(convo_agent, "Hello agent!", user_id, session_id)
        run_conversation_turn(convo_agent, "I am planning to learn a new programming language. I am thinking about Python.", user_id, session_id)
        run_conversation_turn(convo_agent, "The weather today is quite sunny and pleasant.", user_id, session_id)

        # Agent should recall context about sunny weather
        run_conversation_turn(convo_agent, "What do you know about the current weather?", user_id, session_id)

        # Agent should recall context about Python
        run_conversation_turn(convo_agent, "Any thoughts on Python?", user_id, session_id)

        run_conversation_turn(convo_agent, "Remember this: all project meetings are at 3 PM on Fridays.", user_id, session_id)

        # Agent should try to recall the meeting time from context
        run_conversation_turn(convo_agent, "What was that important thing I asked you to remember about meetings?", user_id, session_id)

    except Exception as e:
        print(f"An error occurred during the conversational example: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore the original get_llm_provider function if it was patched
        if original_get_llm_provider:
            llm_providers.get_llm_provider = original_get_llm_provider
        elif hasattr(llm_providers, "get_llm_provider"): # Clean up our mock patch if original didn't exist
             delattr(llm_providers, "get_llm_provider")


        # Clean up the temporary memory store directory
        if example_store_base_path.exists():
            shutil.rmtree(example_store_base_path)
        print(f"\nCleaned up temporary store: {example_store_base_path}")

```
