import json
from typer.testing import CliRunner
from gist_memory.cli import app
from gist_memory.embedding_pipeline import MockEncoder
import pytest


@pytest.fixture(autouse=True)
def use_mock_encoder(monkeypatch):
    enc = MockEncoder()
    monkeypatch.setattr(
        "gist_memory.embedding_pipeline._load_model", lambda *a, **k: enc
    )
    yield


def test_cli_init_add_query(tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, ["init", str(tmp_path), "--agent-name", "tester"])
    assert result.exit_code == 0

    result = runner.invoke(
        app,
        ["add", "--agent-name", str(tmp_path), "--text", "hello world"],
    )
    assert result.exit_code == 0

    result = runner.invoke(
        app,
        [
            "query",
            "--agent-name",
            str(tmp_path),
            "--query-text",
            "hello",
            "--json",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout.strip())
    assert data["memories"]


def test_cli_stats(tmp_path):
    runner = CliRunner()
    runner.invoke(app, ["init", str(tmp_path), "--agent-name", "tester"])
    runner.invoke(app, ["add", "--agent-name", str(tmp_path), "--text", "alpha"])
    result = runner.invoke(app, ["stats", str(tmp_path), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout.strip())
    assert data["prototypes"] == 1
