import importlib
from types import SimpleNamespace

import compact_memory.__main__ as main_mod


def test_main_invokes_cli_without_args(monkeypatch):
    called = {}
    monkeypatch.setattr(main_mod, "sys", SimpleNamespace(argv=["compact-memory"]))
    importlib.reload(main_mod)
    cli_mod = importlib.import_module("compact_memory.cli")
    monkeypatch.setattr(cli_mod, "app", lambda: called.setdefault("cli", True))
    main_mod.main([])
    assert called == {"cli": True}


def test_main_uses_cli_with_args(monkeypatch):
    called = {}
    monkeypatch.setattr(
        main_mod, "sys", SimpleNamespace(argv=["compact-memory", "query"])
    )
    importlib.reload(main_mod)
    cli_mod = importlib.import_module("compact_memory.cli")
    monkeypatch.setattr(cli_mod, "app", lambda: called.setdefault("cli", True))
    main_mod.main(["query"])
    assert called == {"cli": True}
