"""
Example script demonstrating the use of ReadAgentGistEngine with an LLMProvider.

This example showcases how to set up the ReadAgentGistEngine with LocalTransformersProvider,
which utilizes a locally running transformer model for gisting, lookup, and QA tasks.

Note:
- This script requires the `transformers` and `torch` (or `tensorflow`) libraries to be installed.
- You will also need a local transformer model compatible with Hugging Face's
  `pipeline` for "text-generation" or "summarization" tasks (depending on the provider's needs).
  `LocalTransformersProvider` by default attempts to use `distilgpt2` if no model is specified
  in its own constructor, but `ReadAgentGistEngine` also specifies default model names in its config.
  Ensure the model used can handle the types of prompts ReadAgent sends.
- Running LLM inferences locally can be resource-intensive (CPU, RAM, GPU if available).
"""

from compact_memory.engines.ReadAgent.engine import ReadAgentGistEngine
from compact_memory.llm_providers import LocalTransformersProvider
from compact_memory.llm_providers_abc import LLMProvider  # For type hinting


def run_readagent_example():
    print("Starting ReadAgentGistEngine with LocalTransformersProvider example...")

    # 1. Initialize an LLMProvider
    # Using LocalTransformersProvider here.
    # You can specify a model_name if you don't want the default ('distilgpt2' for LocalChatModel).
    # For ReadAgent, specific models for gist, lookup, and QA can be passed in engine's config.
    try:
        # Note: LocalChatModel (default for LocalTransformersProvider) may not be ideal for all ReadAgent tasks.
        # A more sophisticated setup might involve different model configurations or a custom provider.
        llm_provider: LLMProvider = LocalTransformersProvider()
        print("LocalTransformersProvider initialized.")
    except ImportError:
        print(
            "Error: LocalTransformersProvider requires 'transformers' and 'torch' (or 'tensorflow')."
        )
        print("Please install them: pip install transformers torch")
        return
    except Exception as e:
        print(
            f"Error initializing LocalTransformersProvider (is a model available?): {e}"
        )
        print("You might need to download a model like 'distilgpt2' or specify one.")
        return

    # 2. Configure and initialize ReadAgentGistEngine
    # These model names should be compatible with the llm_provider.
    # For LocalTransformersProvider using Hugging Face pipeline, ensure these models are available.
    engine_config = {
        "gist_length": 60,  # Max new tokens for each gist
        "gist_model_name": "distilgpt2",  # Or another model suitable for summarization/gisting
        "lookup_model_name": "distilgpt2",  # Or another model suitable for understanding context
        "qa_model_name": "distilgpt2",  # Or another model suitable for QA
        "lookup_max_tokens": 30,  # Max new tokens for the lookup response
        "qa_max_new_tokens": 150,  # Max new tokens for the final QA answer
    }

    read_agent_engine = ReadAgentGistEngine(
        llm_provider=llm_provider, config=engine_config
    )
    print("ReadAgentGistEngine initialized with LocalTransformersProvider.")

    # 3. Prepare some text to process
    # Using a simple multi-episode text.
    document_text = (
        "Episode 1: The history of Compact Memory project.\n"
        "Compact Memory started as an effort to manage context windows in LLMs effectively. "
        "It explores various techniques for text compression and summarization.\n\n"
        "Episode 2: Core components of Compact Memory.\n"
        "The library includes base classes for compression engines, chunkers, "
        "and a registry for managing different engines. It also provides utilities for token counting.\n\n"
        "Episode 3: Using ReadAgentGistEngine.\n"
        "The ReadAgentGistEngine is a specific engine that creates gists of text episodes "
        "and uses them for summarization or to inform question-answering processes."
    )
    print(f"\nOriginal Document Text:\n{document_text}\n")

    # 4. Perform a Question Answering task
    question = "What is the ReadAgentGistEngine?"
    # The llm_token_budget here is for the final answer's character length truncation (a fallback).
    # The actual token generation limit for the LLM QA call is qa_max_new_tokens.
    answer_budget_chars = 200

    print(f"Performing QA task with question: '{question}'")
    try:
        compressed_answer, trace_qa = read_agent_engine.compress(
            document_text, llm_token_budget=answer_budget_chars, query=question
        )
        print(f"\n--- QA Result ---")
        print(f"Question: {question}")
        print(f"Answer: {compressed_answer.text}")
        # print(f"\nQA Trace:")
        # for step in trace_qa.steps:
        #     print(f"  - Type: {step['type']}, Details: {step['details']}")
        print(f"--- End QA Result ---")

    except Exception as e:
        print(f"\nError during ReadAgent QA processing: {e}")
        print(
            "This could be due to issues with the local model, resource limits, or unexpected model output."
        )
        print(
            "Consider using a smaller model or ensuring your environment is correctly set up."
        )

    # 5. Perform a Summarization task (no query)
    summary_budget_chars = (
        300  # Character budget for the final summary (used for truncation)
    )
    print(f"\nPerforming Summarization task...")
    try:
        compressed_summary, trace_summary = read_agent_engine.compress(
            document_text, llm_token_budget=summary_budget_chars
        )
        print(f"\n--- Summarization Result ---")
        print(f"Summary: {compressed_summary.text}")
        # print(f"\nSummarization Trace:")
        # for step in trace_summary.steps:
        #     print(f"  - Type: {step['type']}, Details: {step['details']}")
        print(f"--- End Summarization Result ---")

    except Exception as e:
        print(f"\nError during ReadAgent Summarization processing: {e}")

    # Example with llm_provider=None (simulation mode)
    print(
        "\n--- Example with ReadAgentGistEngine in Simulation Mode (llm_provider=None) ---"
    )
    engine_simulated = ReadAgentGistEngine(llm_provider=None, config=engine_config)
    sim_question = "What is episode 1 about?"
    sim_answer, _ = engine_simulated.compress(
        document_text, llm_token_budget=100, query=sim_question
    )
    print(f"Simulated Question: {sim_question}")
    print(f"Simulated Answer: {sim_answer.text}")

    sim_summary, _ = engine_simulated.compress(document_text, llm_token_budget=150)
    print(f"Simulated Summary: {sim_summary.text}")
    print("--- End Simulation Mode Example ---")

    print("\nReadAgentGistEngine with LocalTransformersProvider example finished.")


if __name__ == "__main__":
    run_readagent_example()
