import builtins
import pytest
from textual.app import App
from gist_memory.tui import _disk_usage, run_tui
from gist_memory.json_npy_store import JsonNpyVectorStore
from gist_memory.embedding_pipeline import MockEncoder


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


def test_wizard_load(monkeypatch, tmp_path):
    enc = MockEncoder()
    monkeypatch.setattr(
        "gist_memory.embedding_pipeline._load_model", lambda *a, **k: enc
    )

    async def autopilot(pilot):
        await pilot.pause(0.1)
        await pilot.press("l")
        await pilot.pause(0.2)
        pilot.app.exit()

    orig_run = App.run

    def patched_run(self, *a, **kw):
        return orig_run(self, headless=True, auto_pilot=autopilot)

    monkeypatch.setattr(App, "run", patched_run)

    run_tui(str(tmp_path))
    store = JsonNpyVectorStore(str(tmp_path))
    assert len(store.memories) == 4
    assert len(store.prototypes) == 3


def _patch_mock_encoder(monkeypatch):
    enc = MockEncoder()
    monkeypatch.setattr(
        "gist_memory.embedding_pipeline._load_model", lambda *a, **k: enc
    )
    return enc


def _patch_run(monkeypatch, autopilot):
    orig_run = App.run

    def patched_run(self, *a, **kw):
        return orig_run(self, headless=True, auto_pilot=autopilot)

    monkeypatch.setattr(App, "run", patched_run)


def test_wizard_create_exit_no(monkeypatch, tmp_path):
    _patch_mock_encoder(monkeypatch)

    async def autopilot(pilot):
        await pilot.press("c")
        await pilot.press("h", "i")
        await pilot.press("enter")
        await pilot.press("f5")
        await pilot.press("q")
        await pilot.pause(0.1)
        await pilot.press("n")
        await pilot.exit(None)

    _patch_run(monkeypatch, autopilot)
    run_tui(str(tmp_path))
    store = JsonNpyVectorStore(str(tmp_path))
    assert len(store.memories) == 1
    assert len(store.prototypes) == 1
    assert not (tmp_path.with_suffix(".zip")).exists()


def test_wizard_create_exit_yes(monkeypatch, tmp_path):
    _patch_mock_encoder(monkeypatch)

    async def autopilot(pilot):
        await pilot.press("c")
        await pilot.press("a")
        await pilot.press("enter")
        await pilot.press("f5")
        await pilot.press("q")
        await pilot.pause(0.1)
        await pilot.press("y")
        await pilot.exit(None)

    _patch_run(monkeypatch, autopilot)
    run_tui(str(tmp_path))
    store = JsonNpyVectorStore(str(tmp_path))
    assert len(store.memories) == 1
    assert len(store.prototypes) == 1
    assert (tmp_path.with_suffix(".zip")).exists()


def test_talk_mode_llm(monkeypatch, tmp_path):
    _patch_mock_encoder(monkeypatch)

    prompts = {}

    class Dummy:
        def __init__(self, *a, **kw):
            pass

        def reply(self, text):
            prompts["text"] = text
            return "resp"

    monkeypatch.setattr("gist_memory.local_llm.LocalChatModel", Dummy)

    async def autopilot(pilot):
        await pilot.press("c")
        await pilot.press("h", "i")
        await pilot.press("enter")
        await pilot.press("q")
        await pilot.pause(0.1)
        await pilot.press("n")
        await pilot.exit(None)

    _patch_run(monkeypatch, autopilot)
    run_tui(str(tmp_path))

    assert "hi" in prompts.get("text", "")


def test_install_models_command(monkeypatch, tmp_path):
    _patch_mock_encoder(monkeypatch)

    calls = []

    def dummy_embed(name):
        calls.append(name)
        class Dummy:
            pass
        return Dummy()

    def dummy_from_pretrained(name, **kw):
        calls.append(name)
        return Dummy()

    class Dummy:
        pass

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", dummy_embed)
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", dummy_from_pretrained
    )
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained", dummy_from_pretrained
    )

    async def autopilot(pilot):
        await pilot.press("c")
        await pilot.press(
            "/", "i", "n", "s", "t", "a", "l", "l", "-", "m", "o", "d", "e", "l", "s"
        )
        await pilot.press("enter")
        await pilot.press("q")
        await pilot.pause(0.1)
        await pilot.press("n")
        await pilot.exit(None)

    _patch_run(monkeypatch, autopilot)
    run_tui(str(tmp_path))

    assert "all-MiniLM-L6-v2" in calls
    assert "distilgpt2" in calls


def test_tui_logging(monkeypatch, tmp_path):
    _patch_mock_encoder(monkeypatch)

    async def autopilot(pilot):
        await pilot.press("c")
        await pilot.press("h", "i")
        await pilot.press("enter")
        await pilot.press(
            "/", "l", "o", "g", " ", "l", "o", "g", ".", "t", "x", "t"
        )
        await pilot.press("enter")
        await pilot.press("/", "q", "u", "e", "r", "y", " ", "h", "i")
        await pilot.press("enter")
        await pilot.press("q")
        await pilot.pause(0.1)
        await pilot.press("n")
        await pilot.exit(None)

    _patch_run(monkeypatch, autopilot)
    run_tui(str(tmp_path))

    log_path = tmp_path / "log.txt"
    assert log_path.exists()
    assert log_path.read_text() != ""
