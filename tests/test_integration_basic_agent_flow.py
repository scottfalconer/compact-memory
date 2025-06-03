import pytest
from pathlib import Path
import shutil
import os

from compact_memory.api_config import CompactMemoryConfig, EmbeddingConfig, ChunkerConfig, MemoryStoreConfig, StrategyConfig
from compact_memory.new_agent import CompactMemoryAgent
from compact_memory.api_models import IngestionReport, CompressedMemoryContext, AgentInteractionResponse

DEFAULT_TEST_EMBEDDING_MODEL = os.environ.get("CM_TEST_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

@pytest.fixture
def test_store_path(tmp_path: Path) -> Path:
    path = tmp_path / "integration_agent_store"
    if path.exists():
        shutil.rmtree(path)
    # Path is created by the agent's __init__ for JsonNpyVectorStore or when store is saved.
    # No need to mkdir here, let the agent/store handle it.
    return path

@pytest.fixture
def basic_agent_config(test_store_path: Path) -> CompactMemoryConfig:
    return CompactMemoryConfig(
        default_embedding_config=EmbeddingConfig(
            provider="huggingface",
            model_name=DEFAULT_TEST_EMBEDDING_MODEL
        ),
        default_chunker_config=ChunkerConfig(
            type="sentence_window",
            params={"window_size": 1, "overlap": 0} # Each sentence is a chunk
        ),
        memory_store_config=MemoryStoreConfig(
            type="default_json_npy",
            path=str(test_store_path),
            params={} # Ensure params is a dict
        ),
        strategies={
            "first_last_test": StrategyConfig(
                id="first_last_test_instance",
                strategy_class_id="FirstLastStrategy",
                params={"first_k": 1, "last_k": 1}
            )
        },
        default_tokenizer_name=None,
        default_llm_provider_config=None
    )

def test_basic_agent_flow_init_ingest_retrieve(basic_agent_config: CompactMemoryConfig, test_store_path: Path):
    agent = None
    try:
        # This agent will use real components: HuggingFace embedder, SentenceWindowChunker, JsonNpyStore, FirstLastStrategy
        agent = CompactMemoryAgent(config=basic_agent_config, default_strategy_id="first_last_test")
        assert agent is not None
        # JsonNpyVectorStore creates its directory on init if it doesn't exist
        assert test_store_path.exists()
        assert agent.storage_path == test_store_path
        assert "first_last_test" in agent.strategies # Check that strategy instance id is key

        text_to_ingest = "This is sentence one. This is sentence two. This is sentence three. This is sentence four. This is sentence five."
        # SentenceWindowChunker with window_size=1, overlap=0 should give 5 chunks
        ingest_report = agent.ingest(text=text_to_ingest, metadata={"doc_id": "test_doc_01"}, user_id="integ_user")

        assert isinstance(ingest_report, IngestionReport)
        assert ingest_report.status == "success"
        assert ingest_report.items_processed == 5
        assert len(ingest_report.item_ids) == 5

        # Check if store files are created (specific to JsonNpyVectorStore)
        assert (test_store_path / "memories.json").exists()
        assert (test_store_path / "vectors.npy").exists()
        assert (test_store_path / "meta.json").exists()

        query = "A query related to the ingested text."
        retrieved_context = agent.retrieve_context(
            query=query,
            strategy_id="first_last_test", # Use instance ID
            budget=50, # Budget in tokens for FirstLastStrategy usually means number of chunks
            user_id="integ_user"
        )

        assert isinstance(retrieved_context, CompressedMemoryContext)
        assert retrieved_context.strategy_id_used == "first_last_test_instance" # StrategyConfig.id is used here

        compressed_content = retrieved_context.compressed_text.lower().strip()
        # FirstLastStrategy with first_k=1, last_k=1 should combine first and last chunks
        expected_combined_text = "this is sentence one. this is sentence five."
        assert expected_combined_text == compressed_content

        # Check source references from FirstLastStrategy
        # It should return references to the first and last original chunks.
        if ingest_report.items_processed >= 2: # Need at least 2 chunks for first and last
            assert len(retrieved_context.source_references) == 2
            first_ref_text = retrieved_context.source_references[0].text_snippet.strip().lower()
            last_ref_text = retrieved_context.source_references[-1].text_snippet.strip().lower()
            assert "this is sentence one." == first_ref_text
            assert "this is sentence five." == last_ref_text

        # Test process_message without LLM response generation
        interaction_response = agent.process_message(
            message=query,
            user_id="integ_user",
            generate_response=False,
            retrieval_strategy_id="first_last_test" # Use instance ID
        )
        assert isinstance(interaction_response, AgentInteractionResponse)
        assert interaction_response.llm_response is None
        assert interaction_response.error_message is None
        assert interaction_response.context_used.compressed_text == retrieved_context.compressed_text
        assert interaction_response.context_used.strategy_id_used == "first_last_test_instance"

    finally:
        # Teardown: close store and remove directory
        if agent and hasattr(agent, 'store') and agent.store:
            if hasattr(agent.store, 'close') and callable(agent.store.close):
                 agent.store.close()
            # If store needs explicit deletion of files or has a clear method
            if hasattr(agent.store, 'delete_store_files') and callable(agent.store.delete_store_files):
                agent.store.delete_store_files()

        if test_store_path.exists():
            shutil.rmtree(test_store_path)


from compact_memory.api_functions import compress_text
# CompressedMemoryContext is already imported
from compact_memory.token_utils import get_tokenizer
# No need to import DEFAULT_TEST_EMBEDDING_MODEL for this specific test

def test_compress_text_integration_first_last_strategy():
    # Tests the compress_text stateless function with the real FirstLastStrategy.
    text_with_newlines = "Sentence one.\nSentence two.\nSentence three.\nSentence four.\nSentence five."

    try:
        tokenizer = get_tokenizer("gpt2")
    except Exception as e:
        pytest.skip(f"Could not load gpt2 tokenizer for test_compress_text: {e}")

    strategy_class_id = "FirstLastStrategy"
    budget = 100 # Example token budget, actual token count depends on tokenizer and output
    # Select first 2 lines, last 1 line, use " :: " as separator between first_k and last_k blocks
    strategy_params = {"first_k": 2, "last_k": 1, "separator": " :: "}

    llm_provider_instance = None # Not needed for FirstLastStrategy

    result_context = compress_text(
        text=text_with_newlines,
        strategy_class_id=strategy_class_id,
        budget=budget,
        strategy_params=strategy_params,
        tokenizer_instance=tokenizer,
        llm_provider_instance=llm_provider_instance
    )

    assert isinstance(result_context, CompressedMemoryContext)
    assert result_context.strategy_id_used == "FirstLastStrategy"

    assert "Sentence one." in result_context.compressed_text
    assert "Sentence two." in result_context.compressed_text
    assert "Sentence five." in result_context.compressed_text
    assert "Sentence three." not in result_context.compressed_text
    assert "Sentence four." not in result_context.compressed_text
    assert result_context.compressed_text.count(" :: ") == 1

    assert len(result_context.source_references) == 3
    assert result_context.source_references[0].text_snippet.strip() == "Sentence one."
    assert result_context.source_references[1].text_snippet.strip() == "Sentence two."
    assert result_context.source_references[2].text_snippet.strip() == "Sentence five."

    assert result_context.budget_info is not None
    assert result_context.budget_info["requested_budget"] == budget
    assert "final_tokens" in result_context.budget_info

    assert result_context.full_trace is not None
    # The actual strategy_name in trace might be the class ID or a more specific name.
    # For FirstLastStrategy, it's typically its class name.
    assert result_context.full_trace["strategy_name"] == "FirstLastStrategy"
    assert result_context.full_trace["original_tokens"] is not None
    assert result_context.full_trace["compressed_tokens"] is not None
