# --- RAG Pipeline Example using CompactMemoryAgent ---

from compact_memory import (
    CompactMemoryConfig,
    EmbeddingConfig,
    ChunkerConfig,
    MemoryStoreConfig,
    StrategyConfig,
    CompactMemoryAgent,
    # LLMProviderAPIConfig # Not strictly needed for this example's agent if it only retrieves
)
from pathlib import Path
import shutil
import os # For environment variable

# Assume a simple LLM provider mock or a real lightweight one for the app's final response generation
# This example focuses on Compact Memory's role in context retrieval.

# Use a common default embedding model for tests, allow override via environment variable
DEFAULT_TEST_EMBEDDING_MODEL = os.environ.get("CM_TEST_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def setup_rag_agent(store_base_path: Path) -> CompactMemoryAgent:
    """Helper function to configure and initialize a CompactMemoryAgent for RAG."""
    store_path = store_base_path / "rag_store"
    if store_path.exists(): # Clean up for repeatable example
        shutil.rmtree(store_path)
    # The agent's JsonNpyVectorStore will create the directory.

    rag_config = CompactMemoryConfig(
        default_embedding_config=EmbeddingConfig(
            provider="huggingface",
            model_name=DEFAULT_TEST_EMBEDDING_MODEL
        ),
        default_chunker_config=ChunkerConfig(
            type="sentence_window",
            params={"window_size": 5, "overlap": 1} # Larger window for more context per chunk
        ),
        memory_store_config=MemoryStoreConfig(
            type="default_json_npy",
            path=str(store_path),
            params={} # Ensure params is a dict
        ),
        strategies={
            "rag_retrieval": StrategyConfig(
                id="rag_retrieval_instance", # Name of this strategy configuration instance
                strategy_class_id="FirstKStrategy", # Registered ID of the strategy class
                params={"k": 3} # Retrieve top 3 chunks.
                                # Note: For a production RAG, you'd typically use a strategy that performs
                                # vector similarity search against the query. FirstKStrategy is used here
                                # for simplicity, assuming chunks are somewhat ordered by relevance
                                # or that the store's query method (if advanced) already sorts by relevance.
                                # If the store returns chunks sorted by similarity to the query embedding,
                                # FirstKStrategy effectively becomes "TopKSimilarityStrategy".
            )
        },
        # No agent-level LLM or tokenizer needed if the agent is only used for retrieval
        # and strategies like FirstKStrategy don't require them.
        default_llm_provider_config=None,
        default_tokenizer_name=None,
    )

    # Optionally, save the config for inspection or reuse
    # config_file_path = store_path.parent / "rag_agent_config.yaml" # Save outside store itself
    # rag_config.save_to_file(config_file_path)
    # print(f"RAG Agent configuration saved to: {config_file_path}")

    # Initialize the agent, setting "rag_retrieval" as the default strategy for retrieve_context calls
    agent = CompactMemoryAgent(config=rag_config, default_strategy_id="rag_retrieval")
    print(f"RAG Agent initialized. Memory store location: {agent.storage_path}")
    return agent

def populate_knowledge_base(agent: CompactMemoryAgent, documents: Dict[str, str]):
    """Ingests a dictionary of documents (doc_id: content) into the agent's memory."""
    print("\nPopulating knowledge base...")
    for doc_id, text_content in documents.items():
        report = agent.ingest(text=text_content, metadata={"doc_id": doc_id}, user_id="knowledge_base_main")
        if report.status == "success":
            print(f"  Ingested '{doc_id}' ({report.items_processed} chunks).")
        else:
            print(f"  Failed to ingest '{doc_id}': {report.message}")

def answer_query_with_rag(
    agent: CompactMemoryAgent,
    query: str,
    # This would be your application's actual LLM call:
    # llm_for_response_generation: YourLLMInterface
    mock_app_llm_response_generator # Placeholder for app's LLM call for this example
    ):
    """Simulates answering a query using RAG with Compact Memory for context retrieval."""
    print(f"\nAnswering query: '{query}'")

    # 1. Retrieve context from Compact Memory Agent
    #    The agent uses its configured default strategy ("rag_retrieval")
    try:
        retrieved_context = agent.retrieve_context(
            query=query,
            # strategy_id="rag_retrieval", # Can be omitted if default_strategy_id is set in agent
            budget=500, # Example budget for the compressed context (e.g., in tokens)
                        # The FirstKStrategy's 'k' parameter primarily controls retrieved content size here.
            user_id="knowledge_base_main" # Assuming KB is not user-specific for retrieval here
        )
        print(f"  Retrieved context using strategy '{retrieved_context.strategy_id_used}'.")
        print(f"  Compressed context (first 200 chars): '{retrieved_context.compressed_text[:200].strip()}...'")
        if retrieved_context.source_references:
            print(f"  Context includes {len(retrieved_context.source_references)} source snippets.")
    except Exception as e:
        print(f"  Error retrieving context from Compact Memory Agent: {e}")
        return "Sorry, I couldn't retrieve the necessary information to answer your query."

    # 2. Construct a prompt for the application's main LLM
    #    This prompt combines the retrieved context with the original query.
    prompt = (
        f"Please answer the following query based *only* on the provided context.\n\n"
        f"Query: {query}\n\n"
        f"Context:\n\"\"\"\n{retrieved_context.compressed_text}\n\"\"\""
    )
    # print(f"\n  Constructed prompt for application's LLM (simplified):\n  '{prompt[:300]}...'") # For debugging

    # 3. Generate the final answer using the application's LLM
    #    In a real application, this would be a call to your chosen LLM (OpenAI, Gemini, local, etc.)
    #    final_answer = llm_for_response_generation.generate(prompt, max_tokens=150)
    final_answer = mock_app_llm_response_generator(prompt, query, retrieved_context.compressed_text) # Using mock
    print(f"\n  Generated final answer (via application's LLM - mocked):\n  '{final_answer}'")
    return final_answer

# --- Example Usage ---
if __name__ == "__main__":
    # Define a temporary path for this example's memory store
    example_store_base_path = Path("./temp_rag_example_store_docs")

    # A simple mock for the application's final LLM call
    def simple_mock_app_llm(full_prompt: str, original_query: str, context_text: str) -> str:
        # Crude check if context was used for specific queries
        if "capital of france" in original_query.lower():
            if "paris" in context_text.lower() and "eiffel tower" in context_text.lower():
                return "Based on the provided context, Paris is the capital of France, known for the Eiffel Tower."
            else:
                return "The provided context does not seem to contain specific information about the capital of France."
        elif "python" in original_query.lower():
            if "programming language" in context_text.lower() and "readability" in context_text.lower():
                return "According to the context, Python is a high-level programming language emphasizing code readability."
            else:
                return "The context does not offer detailed information about Python."
        return f"I have processed your query: '{original_query}'. The retrieved context was: '{context_text[:100]}...'"

    try:
        # 1. Setup the CompactMemoryAgent for RAG
        rag_agent = setup_rag_agent(example_store_base_path)

        # 2. Populate the agent's memory with some documents
        knowledge_base_documents = {
            "doc_france": "Paris is the capital and most populous city of France. It is known for the Eiffel Tower and the Louvre Museum.",
            "doc_python": "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability and its syntax allows programmers to express concepts in fewer lines of code than would be possible in languages such as C++ or Java.",
            "doc_solar": "The solar system consists of the Sun and the astronomical objects gravitationally bound to it, including eight planets."
        }
        populate_knowledge_base(rag_agent, knowledge_base_documents)

        # 3. Simulate answering queries using the RAG pipeline
        print("\n--- Running RAG Queries ---")

        query1 = "What is the capital of France and its famous landmarks?"
        answer1 = answer_query_with_rag(rag_agent, query1, simple_mock_app_llm)

        query2 = "Tell me about Python as a programming language."
        answer2 = answer_query_with_rag(rag_agent, query2, simple_mock_app_llm)

        query3 = "What is the main component of our solar system?" # Should use doc_solar
        answer3 = answer_query_with_rag(rag_agent, query3, simple_mock_app_llm)

    except Exception as e:
        print(f"An error occurred during the RAG example: {e}")
    finally:
        # Clean up the temporary memory store directory
        if example_store_base_path.exists():
            shutil.rmtree(example_store_base_path)
        print(f"\nCleaned up temporary store: {example_store_base_path}")

```
