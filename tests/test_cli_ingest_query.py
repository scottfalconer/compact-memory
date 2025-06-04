from pathlib import Path
from typer.testing import CliRunner

from compact_memory.cli import app
from compact_memory.agent import Agent
from compact_memory.vector_store import InMemoryVectorStore
from compact_memory.embedding_pipeline import get_embedding_dim


runner = CliRunner()


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "COMPACT_MEMORY_COMPACT_MEMORY_PATH": str(tmp_path),
        "COMPACT_MEMORY_DEFAULT_STRATEGY_ID": "none",
        "COMPACT_MEMORY_DEFAULT_MODEL_ID": "tiny-gpt2",
    }


def test_ingest_command_outputs_metrics(tmp_path: Path):
    sample = Path("sample_data/moon_landing/01_landing.txt")
    result = runner.invoke(app, ["ingest", str(sample), "--json"], env=_env(tmp_path))
    assert result.exit_code == 0
    assert "prototype_count" in result.stdout


def test_query_returns_reply(tmp_path: Path, monkeypatch):
    dim = get_embedding_dim()
    store = InMemoryVectorStore(embedding_dim=dim)
    agent = Agent(store)
    agent.add_memory("the sky is blue")
    monkeypatch.setattr("compact_memory.cli._load_agent", lambda path: agent)
    result = runner.invoke(app, ["query", "sky?"], env=_env(tmp_path))
    assert result.exit_code == 0
    assert result.stdout.strip() != ""
