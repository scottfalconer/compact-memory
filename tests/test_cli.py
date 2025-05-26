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
    client = chromadb.EphemeralClient()
    _patch_client(monkeypatch, client)

    runner = CliRunner()
    result = runner.invoke(cli, ["ingest", str(file_path)])
    assert result.exit_code == 0

    store = PrototypeStore(client=client)
    mems = store.dump_memories()
    assert any(m.text == "hello world" for m in mems)


def test_cli_ingest_directory(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "b.txt").write_text("bravo")
    client = chromadb.EphemeralClient()
    _patch_client(monkeypatch, client)

    runner = CliRunner()
    result = runner.invoke(cli, ["ingest", str(tmp_path)])
    assert result.exit_code == 0

    store = PrototypeStore(client=client)
    mems = store.dump_memories()
    texts = {m.text for m in mems}
    assert {"alpha", "bravo"}.issubset(texts)
