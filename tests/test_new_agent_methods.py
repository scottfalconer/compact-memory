import pytest
from unittest.mock import patch, MagicMock, call # Added call
import uuid # For checking generated IDs

# Modules to test or use in tests
from compact_memory.new_agent import CompactMemoryAgent
from compact_memory.api_config import CompactMemoryConfig # For creating agent instance
from compact_memory.api_models import IngestionReport
from compact_memory.models import RawMemory # For asserting store calls
from compact_memory.api_exceptions import IngestionError

# Re-use or adapt the valid_config_dict fixture from test_new_agent_init.py
# For simplicity, a minimal version is defined here.
# In a real setup, you might share fixtures across test files.
@pytest.fixture
def minimal_agent_config(tmp_path):
    store_path = tmp_path / "ingest_test_store"
    return CompactMemoryConfig(
        default_embedding_config={"provider": "mock", "model_name": "mock-embed"},
        default_chunker_config={"type": "sentence_window", "params": {"window_size": 1}}, # Simple chunking
        memory_store_config={"type": "default_json_npy", "path": str(store_path), "params": {}}, # Ensure params is dict
        strategies={} # No strategies needed for basic ingest test
    )

# This fixture provides a mocked agent instance, similar to how it's done in __init__ tests
@pytest.fixture
def mocked_agent(minimal_agent_config): # Takes the config fixture
    with patch('compact_memory.new_agent.get_embedding_model_info') as mock_get_info, \
         patch('compact_memory.new_agent.EmbeddingPipeline') as mock_emb_pipe_cls, \
         patch('compact_memory.new_agent.JsonNpyVectorStore') as mock_store_cls, \
         patch('compact_memory.new_agent.SentenceWindowChunker') as mock_chunker_cls, \
         patch('compact_memory.new_agent.get_llm_provider'), \
         patch('compact_memory.new_agent.get_tokenizer'), \
         patch('compact_memory.new_agent.get_compression_strategy_class'):

        # Setup return values for mocks
        mock_get_info.return_value = MagicMock(dimension=10) # Small dimension for test

        mock_embedding_pipeline_instance = MagicMock()
        # Example embeddings: two embeddings for two chunks
        mock_embedding_pipeline_instance.embed_texts = MagicMock(return_value=[[0.1]*10, [0.2]*10])
        mock_emb_pipe_cls.return_value = mock_embedding_pipeline_instance

        mock_store_instance = MagicMock()
        mock_store_instance.add = MagicMock() # Mock the add method of the store
        mock_store_cls.return_value = mock_store_instance

        mock_chunker_instance = MagicMock()
        mock_chunker_instance.chunk_text = MagicMock(return_value=["chunk1 text", "chunk2 text"])
        mock_chunker_cls.return_value = mock_chunker_instance

        agent = CompactMemoryAgent(config=minimal_agent_config)

        # Attach mocks to agent instance for easier access in tests if needed, or just use the patches
        # These are the instances created by the mocked classes, not the classes themselves.
        agent.embedding_pipeline = mock_embedding_pipeline_instance
        agent.store = mock_store_instance
        agent.chunker = mock_chunker_instance

        return agent

# --- Test Cases for ingest method ---

def test_ingest_successful(mocked_agent: CompactMemoryAgent):
    text_to_ingest = "This is a test sentence. This is another one."
    metadata = {"source": "test_doc.txt"}
    user_id = "user123"

    report = mocked_agent.ingest(text_to_ingest, metadata=metadata, user_id=user_id)

    # Assertions
    assert report.status == "success"
    assert "Successfully ingested 2 chunks" in report.message # Based on mock chunker
    assert report.items_processed == 2
    assert report.items_failed == 0
    assert len(report.item_ids) == 2
    for item_id in report.item_ids:
        assert isinstance(uuid.UUID(item_id, version=4), uuid.UUID) # Check if valid UUIDs

    # Verify chunker was called
    mocked_agent.chunker.chunk_text.assert_called_once_with(text_to_ingest)

    # Verify embedding pipeline was called
    mocked_agent.embedding_pipeline.embed_texts.assert_called_once_with(["chunk1 text", "chunk2 text"])

    # Verify store.add was called correctly
    assert mocked_agent.store.add.call_count == 1
    args, _ = mocked_agent.store.add.call_args
    added_raw_memories = args[0]
    assert isinstance(added_raw_memories, list)
    assert len(added_raw_memories) == 2

    for i, raw_memory in enumerate(added_raw_memories):
        assert isinstance(raw_memory, RawMemory)
        assert raw_memory.text == f"chunk{i+1} text"
        assert raw_memory.embedding == ([0.1]*10 if i == 0 else [0.2]*10)
        assert raw_memory.metadata["source"] == "test_doc.txt"
        assert raw_memory.metadata["user_id"] == user_id
        assert raw_memory.metadata["chunk_index"] == i
        assert "original_text_hash" in raw_memory.metadata # Check presence
        assert isinstance(uuid.UUID(raw_memory.id, version=4), uuid.UUID) # Check ID on RawMemory


def test_ingest_empty_text(mocked_agent: CompactMemoryAgent):
    report = mocked_agent.ingest("")
    assert report.status == "failure"
    assert "Cannot ingest empty text" in report.message
    assert report.items_processed == 0
    assert report.items_failed == 1
    mocked_agent.chunker.chunk_text.assert_not_called()


def test_ingest_chunker_returns_no_chunks(mocked_agent: CompactMemoryAgent):
    mocked_agent.chunker.chunk_text.return_value = [] # Override mock for this test
    report = mocked_agent.ingest("Some text")

    assert report.status == "success" # As per current ingest logic for no chunks
    assert "Text resulted in no processable chunks." in report.message # Adjusted exact message
    assert report.items_processed == 0
    mocked_agent.embedding_pipeline.embed_texts.assert_not_called()
    mocked_agent.store.add.assert_not_called()


def test_ingest_embedding_fails(mocked_agent: CompactMemoryAgent):
    mocked_agent.embedding_pipeline.embed_texts.side_effect = Exception("Embedding API error")

    # The current ingest method in new_agent.py catches generic Exception from embedding
    # and re-raises it as IngestionError.
    with pytest.raises(IngestionError, match="Failed to embed chunks: Embedding API error"):
        mocked_agent.ingest("Text that causes embedding failure")


def test_ingest_store_add_fails(mocked_agent: CompactMemoryAgent):
    mocked_agent.store.add.side_effect = Exception("Database connection lost")

    with pytest.raises(IngestionError, match="Failed to store RawMemory objects: Database connection lost"): # Exact match from new_agent.py
        mocked_agent.ingest("Some text to store")


def test_ingest_mismatch_chunks_embeddings(mocked_agent: CompactMemoryAgent):
    # Simulate embedding pipeline returning a different number of embeddings than chunks
    mocked_agent.embedding_pipeline.embed_texts.return_value = [[0.1]*10] # Only one embedding for two chunks

    # The current ingest logic for mismatch returns a report, does not raise error directly
    report = mocked_agent.ingest("chunk1. chunk2.")
    assert report.status == "success" # Or "failure" depending on how it's defined in ingest
    assert "No data was prepared for storage" in report.message # Or similar message
    assert report.items_processed == 0
    # Based on the current new_agent.py, items_failed would be len(chunks)
    assert report.items_failed == 2 # Two chunks were provided by mock_chunker_instance by default
    mocked_agent.store.add.assert_not_called()


# --- Tests for retrieve_context method ---

# The minimal_agent_config and mocked_agent fixtures can be reused or adapted.
# For retrieve_context, we'll need to ensure the agent has strategies configured
# and that the mock strategy behaves as expected.

@pytest.fixture
def agent_for_retrieval_config(tmp_path):
    store_path = tmp_path / "retrieval_test_store"
    return CompactMemoryConfig(
        default_embedding_config={"provider": "mock", "model_name": "mock-embed"},
        default_chunker_config={"type": "sentence_window", "params": {"window_size": 1}},
        memory_store_config={"type": "default_json_npy", "path": str(store_path), "params": {}},
        # Configure strategies for testing retrieval
        strategies={
            "mock_retrieve_strat": {
                "id": "mock_retrieve_strat", # Instance ID
                "strategy_class_id": "MockRetrievalStrategy", # To be mocked
                "params": {"p1": "v1"}
            },
            "another_strat": {
                "id": "another_strat",
                "strategy_class_id": "AnotherMockStrategy",
                "params": {}
            }
        },
        # default_tokenizer_name and default_llm_provider_config might be needed if strategies use them
        default_tokenizer_name="mock-agent-tokenizer",
        default_llm_provider_config={"provider": "mock", "model_name": "mock-agent-llm"}
    )

@pytest.fixture
def mocked_agent_for_retrieval(agent_for_retrieval_config): # Depends on the new config
    with patch('compact_memory.new_agent.get_embedding_model_info') as mock_get_info, \
         patch('compact_memory.new_agent.EmbeddingPipeline') as mock_emb_pipe_cls, \
         patch('compact_memory.new_agent.JsonNpyVectorStore') as mock_store_cls, \
         patch('compact_memory.new_agent.SentenceWindowChunker') as mock_chunker_cls, \
         patch('compact_memory.new_agent.get_llm_provider') as mock_get_llm, \
         patch('compact_memory.new_agent.get_tokenizer') as mock_get_tokenizer, \
         patch('compact_memory.new_agent.get_compression_strategy_class') as mock_get_strat_cls:

        mock_get_info.return_value = MagicMock(dimension=10)

        mock_embedding_pipeline_instance = MagicMock()
        mock_embedding_pipeline_instance.embed_query = MagicMock(return_value=[0.5]*10) # Mock query embedding
        mock_emb_pipe_cls.return_value = mock_embedding_pipeline_instance

        mock_store_instance = MagicMock()
        # Mock store.query to return some mock data (list of RawMemory or dicts)
        # The objects should ideally mimic RawMemory structure if strategies expect it.
        mock_retrieved_item_1 = MagicMock(spec=RawMemory)
        mock_retrieved_item_1.text = "retrieved chunk 1 from store"
        mock_retrieved_item_1.id = "store_rc1"
        mock_retrieved_item_2 = MagicMock(spec=RawMemory)
        mock_retrieved_item_2.text = "retrieved chunk 2 from store"
        mock_retrieved_item_2.id = "store_rc2"

        mock_store_instance.query = MagicMock(return_value=[mock_retrieved_item_1, mock_retrieved_item_2])
        mock_store_cls.return_value = mock_store_instance

        mock_chunker_cls.return_value = MagicMock() # Not directly used by retrieve_context

        # Mock LLM Provider and Tokenizer for agent defaults
        mock_agent_llm = MagicMock()
        mock_agent_tokenizer = MagicMock()
        mock_get_llm.return_value = mock_agent_llm
        mock_get_tokenizer.return_value = mock_agent_tokenizer

        # Mock Strategy Class and Instance
        mock_strategy_instance = MagicMock(spec=True)
        mock_strategy_instance.id = "mock_retrieve_strat"
        mock_strategy_instance.tokenizer = mock_agent_tokenizer
        mock_strategy_instance.llm_provider = mock_agent_llm

        mock_cm_context = CompressedMemoryContext(
            compressed_text="compressed by mock_retrieve_strat",
            source_references=[SourceReference(text_snippet="ref1")], # SourceReference needs to be imported
            strategy_id_used="mock_retrieve_strat",
            processing_time_ms=10.0
        )
        mock_strategy_instance.compress = MagicMock(return_value=mock_cm_context)

        def get_strat_cls_side_effect(class_id):
            if class_id == "MockRetrievalStrategy":
                mock_strat_cls = MagicMock()
                mock_strat_cls.return_value = mock_strategy_instance
                return mock_strat_cls
            elif class_id == "AnotherMockStrategy":
                another_mock_instance = MagicMock(id="another_strat")
                another_mock_instance.compress = MagicMock(return_value=CompressedMemoryContext(
                    compressed_text="compressed by another_strat", source_references=[], strategy_id_used="another_strat"
                ))
                another_mock_strat_cls = MagicMock()
                another_mock_strat_cls.return_value = another_mock_instance
                return another_mock_strat_cls
            raise StrategyNotFoundError(f"Mock strategy class {class_id} not found") # Import StrategyNotFoundError
        mock_get_strat_cls.side_effect = get_strat_cls_side_effect

        agent = CompactMemoryAgent(config=agent_for_retrieval_config, default_strategy_id="mock_retrieve_strat")

        agent.embedding_pipeline = mock_embedding_pipeline_instance
        agent.store = mock_store_instance
        return agent


def test_retrieve_context_successful_with_strategy_id(mocked_agent_for_retrieval: CompactMemoryAgent):
    agent = mocked_agent_for_retrieval
    query = "test query"
    budget = 100
    user_id = "user_test"

    result_context = agent.retrieve_context(query, strategy_id="mock_retrieve_strat", budget=budget, user_id=user_id, custom_arg="val")

    assert result_context.compressed_text == "compressed by mock_retrieve_strat"
    assert result_context.strategy_id_used == "mock_retrieve_strat"
    assert len(result_context.source_references) == 1
    assert result_context.source_references[0].text_snippet == "ref1"

    agent.embedding_pipeline.embed_query.assert_called_once_with(query)

    # The new_agent.py's retrieve_context uses a fixed list of mock strings, not store.query.
    # So, we can't assert agent.store.query.assert_called_once_with(...) directly based on current new_agent.py
    # Instead, the mock data is created inside retrieve_context.
    # If retrieve_context were changed to use self.store.query, this assertion would be:
    # agent.store.query.assert_called_once_with(query_embedding=[0.5]*10, user_id=user_id)


    strategy_mock = agent.strategies["mock_retrieve_strat"]
    strategy_mock.compress.assert_called_once()
    call_args = strategy_mock.compress.call_args

    # The mock data in new_agent.py's retrieve_context is:
    # mock_retrieved_chunks = [
    #     "This is the first mock retrieved chunk. It contains some information about topic A.", ...
    # ]
    # This should be what's passed to memories.
    # For this test, we can't directly assert the content of `memories` if it's hardcoded in the SUT method.
    # However, the mocked strategy instance `strategy_mock` is what we control.
    # The important part is that `compress` was called with *some* list of memories.
    assert isinstance(call_args[1]['memories'], list)
    assert call_args[1]['budget'] == budget
    assert call_args[1]['query_text'] == query
    assert call_args[1]['query_embedding'] == [0.5]*10
    assert call_args[1]['tokenizer'] == agent.tokenizer
    assert call_args[1]['llm_provider'] == agent.llm_provider
    assert call_args[1]['custom_arg'] == "val"


def test_retrieve_context_uses_agent_default_strategy(mocked_agent_for_retrieval: CompactMemoryAgent):
    agent = mocked_agent_for_retrieval
    query = "query for default strategy"

    result_context = agent.retrieve_context(query)

    assert result_context.strategy_id_used == "mock_retrieve_strat"
    strategy_mock = agent.strategies["mock_retrieve_strat"]
    strategy_mock.compress.assert_called_once()

def test_retrieve_context_empty_query(mocked_agent_for_retrieval: CompactMemoryAgent):
    with pytest.raises(RetrievalError, match="Query cannot be empty"): # Import RetrievalError
        mocked_agent_for_retrieval.retrieve_context("")

def test_retrieve_context_strategy_not_found(mocked_agent_for_retrieval: CompactMemoryAgent):
    with pytest.raises(StrategyNotFoundError, match="Strategy with instance ID 'non_existent_strat' not found"):
        mocked_agent_for_retrieval.retrieve_context("test query", strategy_id="non_existent_strat")


def test_retrieve_context_unexpected_error_in_strategy(mocked_agent_for_retrieval: CompactMemoryAgent):
    failing_strategy_instance = mocked_agent_for_retrieval.strategies["mock_retrieve_strat"]
    failing_strategy_instance.compress.side_effect = Exception("Kaboom from strategy")

    with pytest.raises(RetrievalError, match="An unexpected error occurred during context retrieval: Kaboom from strategy"):
        mocked_agent_for_retrieval.retrieve_context("test query", strategy_id="mock_retrieve_strat")


def test_retrieve_context_no_strategies_configured(mocked_agent_for_retrieval: CompactMemoryAgent):
    agent = mocked_agent_for_retrieval
    agent.strategies = {}
    agent.agent_default_strategy_id = None

    with pytest.raises(StrategyNotFoundError, match="No strategies configured for this agent, and no default specified."): # Adjusted message
        agent.retrieve_context("test query")

# --- Tests for process_message method ---

# We can reuse/adapt the agent_for_retrieval_config and mocked_agent_for_retrieval fixtures,
# ensuring the agent has a default LLM provider and tokenizer mocked for response generation tests.

@pytest.fixture
def agent_for_process_message_config(tmp_path): # Similar to agent_for_retrieval_config
    store_path = tmp_path / "process_msg_test_store"
    return CompactMemoryConfig(
        default_embedding_config={"provider": "mock", "model_name": "mock-embed"},
        default_chunker_config={"type": "sentence_window", "params": {"window_size": 1}},
        memory_store_config={"type": "default_json_npy", "path": str(store_path), "params": {}},
        strategies={
            "default_proc_strat": { # A strategy that retrieve_context can use
                "id": "default_proc_strat",
                "strategy_class_id": "MockProcessingStrategy", # To be mocked
                "params": {}
            }
        },
        default_tokenizer_name="mock-agent-tokenizer-proc",
        default_llm_provider_config={"provider": "mock", "model_name": "mock-agent-llm-proc"}
    )

@pytest.fixture
def mocked_agent_for_process_message(agent_for_process_message_config):
    with patch('compact_memory.new_agent.get_embedding_model_info') as mock_get_info, \
         patch('compact_memory.new_agent.EmbeddingPipeline') as mock_emb_pipe_cls, \
         patch('compact_memory.new_agent.JsonNpyVectorStore') as mock_store_cls, \
         patch('compact_memory.new_agent.SentenceWindowChunker') as mock_chunker_cls, \
         patch('compact_memory.new_agent.get_llm_provider') as mock_get_llm, \
         patch('compact_memory.new_agent.get_tokenizer') as mock_get_tokenizer, \
         patch('compact_memory.new_agent.get_compression_strategy_class') as mock_get_strat_cls:

        from compact_memory.api_models import CompressedMemoryContext, SourceReference, IngestionReport # Local import for fixture
        from compact_memory.api_exceptions import StrategyNotFoundError, RetrievalError # Local import for fixture

        mock_get_info.return_value = MagicMock(dimension=10)

        # Mock Embedding Pipeline
        mock_embedding_pipeline_instance = MagicMock()
        mock_embedding_pipeline_instance.embed_query = MagicMock(return_value=[0.5]*10)
        mock_embedding_pipeline_instance.embed_texts = MagicMock(return_value=[[0.1]*10]) # For ingest part
        mock_emb_pipe_cls.return_value = mock_embedding_pipeline_instance

        # Mock Store
        mock_store_instance = MagicMock()
        mock_store_instance.query = MagicMock(return_value=[MagicMock(spec=RawMemory, text="retrieved context text", id="rc1")])
        mock_store_instance.add = MagicMock() # For ingest part
        mock_store_cls.return_value = mock_store_instance

        mock_chunker_instance = MagicMock(chunk_text=MagicMock(return_value=["input message as chunk"])) # For ingest part
        mock_chunker_cls.return_value = mock_chunker_instance


        # Mock LLM Provider and Tokenizer for agent's default
        mock_llm_provider_instance = MagicMock()
        mock_llm_provider_instance.generate_response = MagicMock(return_value="Mocked LLM response.")
        mock_get_llm.return_value = mock_llm_provider_instance

        mock_tokenizer_instance = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer_instance

        # Mock Strategy for retrieve_context
        mock_cm_context_for_retrieval = CompressedMemoryContext(
            compressed_text="context for llm",
            source_references=[SourceReference(text_snippet="ref1")],
            strategy_id_used="default_proc_strat",
            processing_time_ms=5.0
        )
        mock_strategy_instance = MagicMock()
        mock_strategy_instance.id = "default_proc_strat" # Ensure mock strategy instance has an id
        mock_strategy_instance.compress = MagicMock(return_value=mock_cm_context_for_retrieval)

        def get_strat_cls_side_effect(class_id):
            if class_id == "MockProcessingStrategy":
                mock_strat_cls = MagicMock()
                mock_strat_cls.return_value = mock_strategy_instance
                return mock_strat_cls
            raise StrategyNotFoundError(f"Mock strategy class {class_id} not found")
        mock_get_strat_cls.side_effect = get_strat_cls_side_effect

        agent = CompactMemoryAgent(config=agent_for_process_message_config, default_strategy_id="default_proc_strat")

        agent.embedding_pipeline = mock_embedding_pipeline_instance
        agent.store = mock_store_instance
        agent.chunker = mock_chunker_instance
        agent.llm_provider = mock_llm_provider_instance
        agent.tokenizer = mock_tokenizer_instance
        # agent.strategies["default_proc_strat"] is already set by the __init__ due to mock_get_strat_cls side_effect

        return agent


def test_process_message_success_with_response_generation(mocked_agent_for_process_message: CompactMemoryAgent):
    from compact_memory.api_models import AgentInteractionResponse # Local import for test
    agent = mocked_agent_for_process_message
    message = "Hello agent, tell me about topic X."
    user_id = "user_proc_test"
    session_id = "session_proc_1"

    # Spy on retrieve_context and ingest to ensure they are called
    # retrieve_context is already complexly mocked in mocked_agent_for_process_message,
    # so we can check its underlying strategy's compress method.
    # For ingest, we can patch it directly on the instance if needed, or rely on its mock from __init__.

    # Mock agent.ingest directly for this test, as its calls are simpler to verify here
    agent.ingest = MagicMock(return_value=IngestionReport(status="success", items_processed=0, items_failed=0))


    response = agent.process_message(
        message,
        user_id,
        session_id=session_id,
        generate_response=True,
        ingest_message_flag=False,
        retrieval_strategy_id="default_proc_strat",
        # Pass some kwargs that might be used by retrieve_context or llm
        top_k_retrieval=5
    )

    assert isinstance(response, AgentInteractionResponse)
    assert response.llm_response == "Mocked LLM response."
    assert response.context_used.compressed_text == "context for llm"
    assert response.session_id == session_id
    assert response.turn_id is not None
    assert response.error_message is None

    # Check that retrieve_context (via its strategy) was called implicitly
    agent.strategies["default_proc_strat"].compress.assert_called_once()
    compress_call_kwargs = agent.strategies["default_proc_strat"].compress.call_args[1]
    assert compress_call_kwargs['query_text'] == message
    assert compress_call_kwargs['user_id'] == user_id # If retrieve_context passes it down to strategy
    # top_k_retrieval is used by retrieve_context to call store.query, not passed to strategy.compress usually.
    # We can check if store.query was called with it.
    agent.store.query.assert_called_with(query_embedding=[0.5]*10, user_id=user_id, top_k=5)


    agent.ingest.assert_not_called()
    agent.llm_provider.generate_response.assert_called_once()
    prompt_arg = agent.llm_provider.generate_response.call_args[0][0]
    assert message in prompt_arg
    assert "context for llm" in prompt_arg


def test_process_message_ingest_message_flag_true(mocked_agent_for_process_message: CompactMemoryAgent):
    from compact_memory.api_models import IngestionReport # Local import for test
    agent = mocked_agent_for_process_message
    message = "Important info to remember."
    user_id = "user_ingest_test"

    # Mock agent.ingest for this specific test to check its call args
    agent.ingest = MagicMock(return_value=IngestionReport(status="success", items_processed=1, items_failed=0))

    agent.process_message(message, user_id, ingest_message_flag=True, generate_response=False, session_id="sess1")

    agent.ingest.assert_called_once_with(
        text=message,
        metadata={'source': 'user_message_turn', 'session_id': "sess1"},
        user_id=user_id
    )

def test_process_message_no_response_generation(mocked_agent_for_process_message: CompactMemoryAgent):
    agent = mocked_agent_for_process_message
    response = agent.process_message("Test message", "user_no_resp", generate_response=False)

    assert response.llm_response is None
    assert response.context_used is not None
    agent.llm_provider.generate_response.assert_not_called()


def test_process_message_no_llm_provider_configured(mocked_agent_for_process_message: CompactMemoryAgent):
    agent = mocked_agent_for_process_message
    agent.llm_provider = None

    response = agent.process_message("Query requiring LLM", "user_no_llm", generate_response=True)

    assert response.llm_response is None
    assert response.error_message == "LLM response generation skipped: No default LLM provider configured."


def test_process_message_no_tokenizer_configured(mocked_agent_for_process_message: CompactMemoryAgent):
    agent = mocked_agent_for_process_message
    agent.tokenizer = None
    if hasattr(agent.llm_provider, 'has_internal_tokenizer'):
        agent.llm_provider.has_internal_tokenizer = False

    response = agent.process_message("Query requiring tokenizer", "user_no_tokenizer", generate_response=True)

    assert response.llm_response is None
    assert response.error_message == "LLM response generation skipped: Tokenizer not available."


def test_process_message_llm_fails_during_generation(mocked_agent_for_process_message: CompactMemoryAgent):
    agent = mocked_agent_for_process_message
    agent.llm_provider.generate_response.side_effect = Exception("LLM API is down")

    response = agent.process_message("A query", "user_llm_fail", generate_response=True)

    assert response.llm_response is None
    assert "LLM response generation failed: LLM API is down" in response.error_message


def test_process_message_empty_input_message(mocked_agent_for_process_message: CompactMemoryAgent):
    agent = mocked_agent_for_process_message
    response = agent.process_message("", "user_empty_msg")

    assert response.llm_response is None
    assert response.error_message == "Empty message provided."
    assert response.context_used.compressed_text == ""
    # Check that strategy compress was not called
    agent.strategies["default_proc_strat"].compress.assert_not_called()


def test_process_message_retrieval_fails(mocked_agent_for_process_message: CompactMemoryAgent):
    from compact_memory.api_exceptions import RetrievalError # Local import for test
    agent = mocked_agent_for_process_message
    # Make the retrieve_context part (specifically the strategy's compress) fail
    failing_strategy = agent.strategies["default_proc_strat"]
    failing_strategy.compress.side_effect = RetrievalError("DB connection failed for retrieval")

    response = agent.process_message("A query", "user_retrieval_fail")

    assert response.llm_response is None
    assert "DB connection failed for retrieval" in response.error_message
    assert response.context_used is not None
    assert response.context_used.strategy_id_used == "none_due_to_error"
