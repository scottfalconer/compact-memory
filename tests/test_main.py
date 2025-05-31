import importlib
from types import SimpleNamespace

import gist_memory.__main__ as main_mod


def test_main_invokes_cli_without_args(monkeypatch):
    called = {}
    monkeypatch.setattr(main_mod, "sys", SimpleNamespace(argv=["gist-memory"]))
    importlib.reload(main_mod)
    cli_mod = importlib.import_module("gist_memory.cli")
    monkeypatch.setattr(cli_mod, "app", lambda: called.setdefault("cli", True))
    main_mod.main([])
    assert called == {"cli": True}


def test_main_uses_cli_with_args(monkeypatch):
    called = {}
    monkeypatch.setattr(
        main_mod, "sys", SimpleNamespace(argv=["gist-memory", "ingest"])
    )
    importlib.reload(main_mod)
    cli_mod = importlib.import_module("gist_memory.cli")
    monkeypatch.setattr(cli_mod, "app", lambda: called.setdefault("cli", True))
    main_mod.main(["ingest"])
    assert called == {"cli": True}
