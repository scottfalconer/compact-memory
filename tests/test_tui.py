import builtins
import pytest
from gist_memory.tui import _disk_usage, run_tui


def test_disk_usage(tmp_path):
    f1 = tmp_path / "a.txt"
    f1.write_text("alpha")
    f2 = tmp_path / "b.txt"
    f2.write_text("beta")
    expected = f1.stat().st_size + f2.stat().st_size
    assert _disk_usage(tmp_path) == expected


def test_run_tui_import_error(monkeypatch, tmp_path):
    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("textual"):
            raise ImportError("no textual")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError):
        run_tui(str(tmp_path))
