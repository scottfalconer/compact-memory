from typer.testing import CliRunner
import pytest

from compact_memory.cli import app

runner = CliRunner()


def test_download_embedding_model_invokes_util(monkeypatch):
    called = {}

    def fake_download(name: str) -> None:
        called["name"] = name

    monkeypatch.setattr(
        "compact_memory.cli.dev_commands.util_download_embedding_model", fake_download
    )
    result = runner.invoke(
        app,
        ["dev", "download-embedding-model", "--model-name", "all-MiniLM-L6-v2"],
    )
    assert result.exit_code == 0, result.stderr
    assert called["name"] == "all-MiniLM-L6-v2"


def test_download_chat_model_invokes_util(monkeypatch):
    called = {}

    def fake_download(name: str) -> None:
        called["name"] = name

    monkeypatch.setattr(
        "compact_memory.cli.dev_commands.util_download_chat_model", fake_download
    )
    result = runner.invoke(
        app, ["dev", "download-chat-model", "--model-name", "tiny-gpt2"]
    )
    assert result.exit_code == 0, result.stderr
    assert called["name"] == "tiny-gpt2"
