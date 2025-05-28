import importlib
from types import SimpleNamespace

import gist_memory.__main__ as main_mod


def test_main_uses_tui_when_no_args(monkeypatch):
    called = {}
    monkeypatch.setattr(main_mod, "sys", SimpleNamespace(argv=["gist-memory"]))
    importlib.reload(main_mod)
    tui_mod = importlib.import_module("gist_memory.tui")
    cli_mod = importlib.import_module("gist_memory.cli")
    monkeypatch.setattr(tui_mod, "run_tui", lambda: called.setdefault("tui", True))
    monkeypatch.setattr(cli_mod, "app", lambda: called.setdefault("cli", True))
    main_mod.main([])
    assert called == {"tui": True}


def test_main_uses_cli_with_args(monkeypatch):
    called = {}
    monkeypatch.setattr(
        main_mod, "sys", SimpleNamespace(argv=["gist-memory", "ingest"])
    )
    importlib.reload(main_mod)
    tui_mod = importlib.import_module("gist_memory.tui")
    cli_mod = importlib.import_module("gist_memory.cli")
    monkeypatch.setattr(tui_mod, "run_tui", lambda: called.setdefault("tui", True))
    monkeypatch.setattr(cli_mod, "app", lambda: called.setdefault("cli", True))
    main_mod.main(["ingest"])
    assert called == {"cli": True}
