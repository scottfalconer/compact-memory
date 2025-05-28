from click.testing import CliRunner

from pathlib import Path

from gist_memory.cli import cli
from gist_memory.store import PrototypeStore


def test_cli_ingest_file(tmp_path):
    file_path = tmp_path / "one.txt"
    file_path.write_text("hello world")

    runner = CliRunner()
    result = runner.invoke(cli, ["--db-path", str(tmp_path), "ingest", str(file_path)])
    assert result.exit_code == 0

    store = PrototypeStore(path=str(tmp_path))
    mems = store.dump_memories()
    assert any(m.text == "hello world" for m in mems)


def test_cli_ingest_directory(tmp_path):
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "b.txt").write_text("bravo")
    runner = CliRunner()
    result = runner.invoke(cli, ["--db-path", str(tmp_path), "ingest", str(tmp_path)])
    assert result.exit_code == 0

    store = PrototypeStore(path=str(tmp_path))
    mems = store.dump_memories()
    texts = {m.text for m in mems}
    assert {"alpha", "bravo"}.issubset(texts)


def test_cli_agentic_ingest(tmp_path):
    path = Path("tests/data/constitution.txt")
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--db-path", str(tmp_path), "--memory-creator", "agentic", "ingest", str(path)],
    )
    assert result.exit_code == 0

    store = PrototypeStore(path=str(tmp_path))
    mems = store.dump_memories()
    assert len(mems) > 1


def test_cli_download_model(monkeypatch):
    called = {}

    def fake_local_embedder(model_name="", local_files_only=True):
        called["name"] = model_name
        called["offline"] = local_files_only
        class Dummy:
            pass
        return Dummy()

    import importlib
    cli_module = importlib.import_module("gist_memory.cli")
    monkeypatch.setattr(cli_module, "LocalEmbedder", fake_local_embedder)

    runner = CliRunner()
    res = runner.invoke(cli, ["download-model", "--model-name", "foo"])
    assert res.exit_code == 0
    assert called == {"name": "foo", "offline": False}
