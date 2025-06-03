# Getting Started with the Compact Memory Python API

This guide will help you get started with the Compact Memory Python API, allowing you to integrate its powerful memory management and text compression features into your Python applications.

## Installation

To install the Compact Memory package, you can use pip:

```bash
pip install compact-memory
```

Some functionalities, especially those involving specific LLM providers (like OpenAI, Gemini, Anthropic) or specialized embedding models, might require you to install extra dependencies. If you plan to use such features, you might need to install them like so:

```bash
# Example for installing with OpenAI and Gemini support (actual extras may vary)
# pip install compact-memory[openai,gemini]
```
*(Note: Check the main documentation for the exact package name and available extras as the project evolves.)*

## Core Concepts

*   **`CompactMemoryAgent`**: This is the primary stateful class you'll interact with. It manages the memory store, orchestrates the ingestion of information, handles context retrieval, and applies compression strategies. It can also integrate with LLMs to generate responses based on the retrieved context.
*   **`CompactMemoryConfig`**: A Pydantic model that holds all the configuration for the `CompactMemoryAgent`. This includes settings for:
    *   **Embedding Models**: How text is converted into numerical vectors.
    *   **Chunkers**: How text is divided into smaller pieces for processing and storage.
    *   **Memory Store**: Where the processed information (text chunks and embeddings) is stored.
    *   **LLM Providers**: (Optional) Connections to Large Language Models for summarization or response generation.
    *   **Tokenizers**: (Optional) Used for counting tokens, often in conjunction with LLMs or specific chunkers/strategies.
    *   **Compression Strategies**: Rules and algorithms for how to compress retrieved context to fit a budget.
*   **Stateless `compress_text` function**: A utility function for quick, on-the-fly text compression using a specified strategy, without the need to set up and manage a persistent `CompactMemoryAgent`.
*   **Strategies**: These are pluggable components that define how context summarization or compression occurs. You can use built-in strategies (like taking the first/last N chunks, summarizing with an LLM) or develop custom ones.

## Basic Usage: `CompactMemoryAgent`

Here's how to configure, initialize, and use the `CompactMemoryAgent`.

### 1. Configuration

First, you need to define a configuration for your agent. This tells the agent which components and settings to use.

```python
from compact_memory import (
    CompactMemoryConfig,
    EmbeddingConfig,
    ChunkerConfig,
    MemoryStoreConfig,
    StrategyConfig,
    LLMProviderAPIConfig # Example if you want to set up a default LLM
    # CompactMemoryAgent will be imported later
)
from pathlib import Path

# Define a path for the memory store (where data will be saved)
store_path = Path("./my_compact_memory_store")
# It's good practice to ensure the parent directory exists if your store doesn't create it.
# JsonNpyVectorStore, for example, will create its own directory.
# store_path.mkdir(parents=True, exist_ok=True) # Usually not needed for JsonNpyVectorStore

config = CompactMemoryConfig(
    default_embedding_config=EmbeddingConfig(
        provider="huggingface", # Using a HuggingFace Sentence Transformer
        model_name="sentence-transformers/all-MiniLM-L6-v2" # A popular, lightweight model
    ),
    default_chunker_config=ChunkerConfig(
        type="sentence_window", # Splits by sentences, then groups them
        params={"window_size": 3, "overlap": 1} # Each chunk has 3 sentences, with 1 sentence overlap
    ),
    memory_store_config=MemoryStoreConfig(
        type="default_json_npy", # Uses the built-in JsonNpyVectorStore
        path=str(store_path),    # Path where the store will save its files
        params={}                # Additional parameters for the store, if any
    ),
    strategies={
        # Define one or more named strategy configurations
        "simple_context": StrategyConfig(
            id="first_last_k_summary", # A unique ID for this specific configuration instance
            strategy_class_id="FirstLastStrategy", # The registered class ID of the strategy
            params={"first_k": 1, "last_k": 1, "separator": " ... "} # Strategy-specific parameters
        )
        # Example of a strategy that might use an LLM (if default_llm_provider_config is set)
        # "llm_summary": StrategyConfig(
        #     id="llm_summarizer_brief",
        #     strategy_class_id="SummarizerStrategy", # Assuming a summarizer strategy exists
        #     params={"max_summary_tokens": 150},
        #     # This strategy might implicitly use the agent's default LLM and tokenizer,
        #     # or you could configure a specific one here using llm_config and tokenizer_name.
        # )
    },
    # Optional: Configure a default LLM provider and tokenizer for the agent and strategies
    # default_llm_provider_config=LLMProviderAPIConfig(
    #     provider="local", # Example: using Ollama or another local provider interface
    #     model_name="llama3", # Replace with your model identifier
    #     # api_key="YOUR_API_KEY_IF_NEEDED",
    #     generation_kwargs={"temperature": 0.5}
    # ),
    # default_tokenizer_name="gpt2", # A common tokenizer, ensure it matches your LLM needs
)

# Optionally, save this configuration to a YAML file for later use or inspection
config_file_path = store_path / "agent_config.yaml"
config.save_to_file(config_file_path)
print(f"Configuration saved to: {config_file_path}")

# Later, you could load it back:
# loaded_config = CompactMemoryConfig.from_file(config_file_path)
# print(f"Loaded configuration version: {loaded_config.version}")
```

### 2. Initializing the Agent

Once you have a configuration object, you can initialize the `CompactMemoryAgent`.

```python
from compact_memory import CompactMemoryAgent
# Assuming 'config' is the CompactMemoryConfig object from the previous step

# Initialize the agent
try:
    # 'simple_context' is the key from config.strategies dict
    agent = CompactMemoryAgent(
        config=config,
        default_strategy_id="simple_context"
    )
    print(f"Agent initialized successfully. Storage path: {agent.storage_path}")
except Exception as e:
    print(f"Error initializing agent: {e}")
    # Handle initialization errors (e.g., model download issues, config errors)
    raise
```

### 3. Ingesting Data

With an initialized agent, you can start ingesting text into its memory.

```python
# Assuming 'agent' is the initialized CompactMemoryAgent

text_data_part1 = "The quick brown fox jumps over the lazy dog. This is a classic sentence used for testing typewriters and fonts."
text_data_part2 = "A stitch in time saves nine. This proverb emphasizes the importance of timely action."
text_data_part3 = "Early to bed and early to rise makes a man healthy, wealthy, and wise. Another piece of timeless advice attributed to Benjamin Franklin."
full_text_data = f"{text_data_part1}\n{text_data_part2}\n{text_data_part3}"

try:
    # Ingest the text. It will be chunked, embedded, and stored.
    report = agent.ingest(
        text=full_text_data,
        metadata={"source": "proverbs_and_sayings.txt", "category": "wisdom"},
        user_id="user_alpha" # Optional: associate data with a user
    )

    print(f"\nIngestion Report:")
    print(f"  Status: {report.status}")
    print(f"  Message: {report.message}")
    print(f"  Items Processed: {report.items_processed}")
    print(f"  Items Failed: {report.items_failed}")
    if report.item_ids:
        print(f"  First few ingested item IDs: {report.item_ids[:3]}")
except Exception as e:
    print(f"Error during ingestion: {e}")
    # Handle ingestion errors (e.g., embedding service unavailable)
```

### 4. Retrieving and Compressing Context

After ingesting data, you can retrieve context relevant to a query. The agent will use the specified (or default) strategy to compress this context.

```python
# Assuming 'agent' is initialized and data has been ingested
query = "What did the fox do?"

try:
    # Retrieve context. If strategy_id is omitted, agent's default_strategy_id is used.
    context = agent.retrieve_context(
        query=query,
        # strategy_id="simple_context", # Explicitly using the strategy instance ID
        budget=50, # Target budget for the compressed context (interpretation depends on strategy)
        user_id="user_alpha" # Optional: retrieve only from this user's data if store supports it
    )

    print(f"\nRetrieved Context (using strategy: {context.strategy_id_used}):")
    print(f"  Compressed Text: '{context.compressed_text}'")
    print(f"  Source References: {len(context.source_references)}")
    if context.source_references:
        print(f"  First source snippet: '{context.source_references[0].text_snippet[:100].strip()}...'")
    print(f"  Processing time (ms): {context.processing_time_ms}")
    if context.budget_info:
        print(f"  Budget Info: {context.budget_info}")

except Exception as e:
    print(f"Error during context retrieval: {e}")
    # Handle retrieval errors (e.g., strategy not found, store query issues)
```

### 5. Processing Messages (with optional LLM response)

The `process_message` method is a versatile way to interact with the agent. It combines context retrieval (using a specified or default strategy) with optional LLM-based response generation. Furthermore, it can simultaneously ingest the user's current message into memory, which is crucial for building conversational history.

**Key parameters for `process_message`:**
*   `message`: The user's input.
*   `user_id`: Identifier for the user.
*   `session_id`: (Optional) Identifier for the ongoing conversation.
*   `generate_response`: (Defaults to `True`) Set to `False` if you only want to retrieve context without generating an LLM reply.
*   `ingest_message_flag`: (Defaults to `False`) Set to `True` to have the current `message` ingested into memory. This is useful for conversational agents that need to remember the history of interaction.
*   `retrieval_strategy_id`: (Optional) Specify which configured strategy instance to use for context retrieval. If omitted, the agent's `default_strategy_id` is used.

**Example without LLM response (context retrieval only):**
This pattern is common in RAG (Retrieval Augmented Generation) pipelines where an external LLM generates the final response based on context retrieved by Compact Memory.

```python
# Assuming 'agent' is initialized
try:
    interaction_response = agent.process_message(
        message="Tell me about the fox and dog again.",
        user_id="user_alpha",
        generate_response=False, # Set to False to only get the context
        retrieval_strategy_id="simple_context" # Specify the strategy instance ID
    )

    print(f"\nProcessed Message (generate_response=False):")
    print(f"  Context Strategy Used: {interaction_response.context_used.strategy_id_used}")
    print(f"  Retrieved Compressed Text: '{interaction_response.context_used.compressed_text}'")
    if interaction_response.error_message:
        print(f"  Error during processing: {interaction_response.error_message}")

except Exception as e:
    print(f"Error processing message: {e}")
```

**Example with LLM response generation:**
To use `generate_response=True`, you must have a default LLM provider configured in your `CompactMemoryConfig` (as shown in the commented-out section of the config example) or ensure the chosen strategy has its own LLM configuration.

```python
# Assuming 'agent' is initialized AND has a default_llm_provider_config set up.
# If not, the following will likely result in an error message within the response.

# For this example to work, uncomment and configure default_llm_provider_config
# in the CompactMemoryConfig earlier.
# Also, ensure the LLM model specified is accessible.

# if agent.llm_provider: # Check if an LLM provider is available
#     try:
#         interaction_with_llm = agent.process_message(
#             message="Summarize the story about the fox.",
#             user_id="user_alpha",
#             generate_response=True, # Agent will try to generate a response
#             retrieval_strategy_id="simple_context"
#         )
#         print(f"\nProcessed Message (generate_response=True):")
#         if interaction_with_llm.llm_response:
#             print(f"  LLM Response: {interaction_with_llm.llm_response}")
#         else:
#             print(f"  LLM Response: None (check error message or agent's LLM config)")
#         print(f"  Context Used: '{interaction_with_llm.context_used.compressed_text}'")
#         if interaction_with_llm.error_message:
#             print(f"  Error: {interaction_with_llm.error_message}")
#     except Exception as e:
#         print(f"Error processing message with LLM: {e}")
# else:
#     print("\nSkipping process_message with LLM response generation as no default LLM provider is configured for the agent.")

```

## Stateless Compression: `compress_text`

For situations where you just need to compress a piece of text using a specific strategy without the overhead of an agent and memory store, use the `compress_text` function.

```python
from compact_memory import compress_text
from compact_memory.token_utils import get_tokenizer # To load a real tokenizer
# LLMProvider and Tokenizer types might be needed for type hinting if you pass them.
# from compact_memory.llm_providers_abc import LLMProvider
# from compact_memory.token_utils import Tokenizer

long_text = ("This is sentence one. This is sentence two. This is sentence three. "
             "This is sentence four. This is sentence five. This is sentence six. "
             "This is sentence seven. This is sentence eight. This is sentence nine. "
             "This is sentence ten.")

try:
    # Strategies often require a tokenizer for budgeting or internal operations.
    # It's good practice to provide one if the strategy might need it.
    tokenizer = get_tokenizer("gpt2") # Load a common tokenizer

    # Using FirstLastStrategy (does not require an LLM)
    compressed_context = compress_text(
        text=long_text,
        strategy_class_id="FirstLastStrategy", # Registered class ID of the strategy
        budget=50,                             # Target budget (e.g., tokens)
        strategy_params={"first_k": 2, "last_k": 2, "separator": " ... "},
        tokenizer_instance=tokenizer,          # Provide the loaded tokenizer
        llm_provider_instance=None             # Not needed for FirstLastStrategy
    )

    print(f"\nStateless Compression Output (strategy: {compressed_context.strategy_id_used}):")
    print(f"  Compressed Text: '{compressed_context.compressed_text}'")

    original_tokens = tokenizer.count_tokens(long_text)
    compressed_tokens = tokenizer.count_tokens(compressed_context.compressed_text)
    print(f"  Original approx tokens: {original_tokens}")
    print(f"  Compressed approx tokens: {compressed_tokens}")
    print(f"  Processing time (ms): {compressed_context.processing_time_ms}")
    if compressed_context.budget_info:
        print(f"  Budget Info: {compressed_context.budget_info}")
    # print(f"  Full trace available: {'Yes' if compressed_context.full_trace else 'No'}")

except Exception as e:
    # This could be due to model download issues for the tokenizer, strategy not found, etc.
    print(f"Error during stateless compression: {e}")
```

## Next Steps

*   Explore the detailed **API Reference** for in-depth information on all classes, methods, and models. (TODO: Add link when available)
*   Dive into practical use cases with these runnable examples:
    *   [RAG Pipeline Example](rag_pipeline_example.py): Demonstrates how to use `CompactMemoryAgent` for the retrieval step in a Retrieval Augmented Generation system.
    *   [Conversational Memory Example](conversational_memory_example.py): Shows how to manage conversational history by ingesting each turn and using it as context for future responses.
    *   [Advanced `compress_text` Usage](advanced_compress_text_example.py): Illustrates how to inspect the detailed output of the stateless `compress_text` function, including source references and compression traces.
*   Learn about developing custom [Compression Strategies](DEVELOPING_COMPRESSION_STRATEGIES.md) to tailor memory compression to your specific needs.
*   Review the [Configuration Details](#) (TODO: Link to a dedicated configuration guide if it exists, or expand on `CompactMemoryConfig` here) to understand all available options for customizing the agent.
```
