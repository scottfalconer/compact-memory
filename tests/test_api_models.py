import pytest
from typing import Dict, Any, List
from pydantic import ValidationError

# Models to test
from compact_memory.api_models import (
    SourceReference,
    CompressedMemoryContext,
    IngestionReport,
    AgentInteractionResponse
)

# --- Fixtures for sample data ---

@pytest.fixture
def sample_source_reference_dict() -> Dict[str, Any]:
    return {
        "document_id": "doc123",
        "chunk_id": "chunk001",
        "text_snippet": "This is a snippet of text.",
        "score": 0.85,
        "metadata": {"source": "test_document.txt"}
    }

@pytest.fixture
def sample_compressed_memory_context_dict(sample_source_reference_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "compressed_text": "This is the compressed summary.",
        "source_references": [sample_source_reference_dict, {"text_snippet": "Another snippet"}], # Test with mixed full/partial refs
        "strategy_id_used": "TestStrategy",
        "budget_info": {"tokens_requested": 100, "tokens_used": 95},
        "processing_time_ms": 120.5,
        "strategy_llm_input": "Prompt to LLM for summarization.",
        "strategy_llm_output": "LLM's summary output."
        # "full_trace" can be tested separately if its structure is complex
    }

@pytest.fixture
def sample_ingestion_report_dict() -> Dict[str, Any]:
    return {
        "status": "success",
        "message": "Ingested 2 items successfully.",
        "items_processed": 2,
        "items_failed": 0,
        "item_ids": ["id_001", "id_002"],
        "processing_time_ms": 50.7
    }

@pytest.fixture
def sample_agent_interaction_response_dict(sample_compressed_memory_context_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "llm_response": "This is the agent's response to the user.",
        "context_used": sample_compressed_memory_context_dict,
        "session_id": "session_abc",
        "turn_id": "turn_123",
        "error_message": None,
        "processing_time_ms": 250.0
    }

# --- Tests for SourceReference ---

def test_source_reference_valid(sample_source_reference_dict: Dict[str, Any]):
    ref = SourceReference(**sample_source_reference_dict)
    assert ref.document_id == sample_source_reference_dict["document_id"]
    assert ref.text_snippet == sample_source_reference_dict["text_snippet"]
    assert ref.score == sample_source_reference_dict["score"]
    assert ref.metadata["source"] == "test_document.txt"

def test_source_reference_minimal(self):
    ref = SourceReference(text_snippet="Minimal snippet.")
    assert ref.text_snippet == "Minimal snippet."
    assert ref.document_id is None
    assert ref.chunk_id is None
    assert ref.score is None
    assert ref.metadata == {} # Default for optional dict if not Field(default_factory=dict)

def test_source_reference_missing_required_fields(self):
    with pytest.raises(ValidationError):
        SourceReference() # text_snippet is required

# --- Tests for CompressedMemoryContext ---

def test_compressed_memory_context_valid(sample_compressed_memory_context_dict: Dict[str, Any]):
    context = CompressedMemoryContext(**sample_compressed_memory_context_dict)
    assert context.compressed_text == sample_compressed_memory_context_dict["compressed_text"]
    assert len(context.source_references) == 2
    assert context.source_references[0].document_id == sample_compressed_memory_context_dict["source_references"][0]["document_id"]
    assert context.source_references[1].text_snippet == "Another snippet"
    assert context.strategy_id_used == "TestStrategy"
    assert context.strategy_llm_input == sample_compressed_memory_context_dict["strategy_llm_input"]

def test_compressed_memory_context_minimal_valid(self):
    # compressed_text, source_references (can be empty list), strategy_id_used are required
    minimal_data = {
        "compressed_text": "Minimal.",
        "source_references": [],
        "strategy_id_used": "MinimalStrat"
    }
    context = CompressedMemoryContext(**minimal_data)
    assert context.compressed_text == "Minimal."
    assert context.budget_info is None # Optional field
    assert context.processing_time_ms is None # Optional

def test_compressed_memory_context_missing_required(self):
    with pytest.raises(ValidationError):
        CompressedMemoryContext(source_references=[], strategy_id_used="Test") # Missing compressed_text

    with pytest.raises(ValidationError):
        CompressedMemoryContext(compressed_text="Text", strategy_id_used="Test") # Missing source_references

# --- Tests for IngestionReport ---

def test_ingestion_report_valid(sample_ingestion_report_dict: Dict[str, Any]):
    report = IngestionReport(**sample_ingestion_report_dict)
    assert report.status == sample_ingestion_report_dict["status"]
    assert report.items_processed == 2
    assert report.item_ids == ["id_001", "id_002"]

def test_ingestion_report_minimal_valid(self):
    # status, items_processed, items_failed are required
    minimal_data = {"status": "partial", "items_processed": 1, "items_failed": 1}
    report = IngestionReport(**minimal_data)
    assert report.status == "partial"
    assert report.message is None
    assert report.item_ids is None

def test_ingestion_report_missing_required(self):
    with pytest.raises(ValidationError):
        IngestionReport(status="success", items_failed=0) # Missing items_processed

# --- Tests for AgentInteractionResponse ---

def test_agent_interaction_response_valid(sample_agent_interaction_response_dict: Dict[str, Any], sample_compressed_memory_context_dict: Dict[str, Any]):
    response = AgentInteractionResponse(**sample_agent_interaction_response_dict)
    assert response.llm_response == sample_agent_interaction_response_dict["llm_response"]
    # Pydantic should handle nested model validation/creation
    assert response.context_used.compressed_text == sample_compressed_memory_context_dict["compressed_text"]
    assert response.session_id == "session_abc"
    assert response.error_message is None

def test_agent_interaction_response_minimal_valid(self, sample_compressed_memory_context_dict: Dict[str, Any]):
    # context_used is required. llm_response is optional.
    minimal_data = {"context_used": sample_compressed_memory_context_dict}
    response = AgentInteractionResponse(**minimal_data)
    assert response.llm_response is None
    assert response.context_used is not None

def test_agent_interaction_response_error_state(self, sample_compressed_memory_context_dict: Dict[str, Any]):
    error_data = {
        "context_used": sample_compressed_memory_context_dict,
        "error_message": "Something went wrong."
    }
    response = AgentInteractionResponse(**error_data)
    assert response.llm_response is None
    assert response.error_message == "Something went wrong."

def test_agent_interaction_response_missing_required(self):
     with pytest.raises(ValidationError):
        AgentInteractionResponse(llm_response="Hi") # Missing context_used
