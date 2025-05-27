import chromadb
from click.testing import CliRunner

from gist_memory.cli import cli
from gist_memory.store import PrototypeStore


def _patch_client(monkeypatch, client):
    import gist_memory.store as store_module
    monkeypatch.setattr(store_module, "default_chroma_client", lambda: client)


def test_cli_ingest_file(tmp_path, monkeypatch):
    file_path = tmp_path / "one.txt"
    file_path.write_text("hello world")
    client = chromadb.PersistentClient(str(tmp_path / "db_file"))
    _patch_client(monkeypatch, client)

    runner = CliRunner()
    result = runner.invoke(cli, ["ingest", str(file_path)])
    assert result.exit_code == 0

    store = PrototypeStore(client=client)
    mems = store.dump_memories()
    assert any(m.text == "hello world" for m in mems)
    # cleanup database
    import shutil
    shutil.rmtree(str(tmp_path / "db_file"))


def test_cli_ingest_directory(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "b.txt").write_text("bravo")
    client = chromadb.PersistentClient(str(tmp_path / "db_file"))
    _patch_client(monkeypatch, client)

    runner = CliRunner()
    result = runner.invoke(cli, ["ingest", str(tmp_path)])
    assert result.exit_code == 0

    store = PrototypeStore(client=client)
    mems = store.dump_memories()
    texts = {m.text for m in mems}
    assert {"alpha", "bravo"}.issubset(texts)
    import shutil
    shutil.rmtree(str(tmp_path / "db_file"))


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
