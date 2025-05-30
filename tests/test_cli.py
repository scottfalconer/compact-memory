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


def test_cli_validate_and_clear(tmp_path):
    runner = CliRunner()
    runner.invoke(app, ["init", str(tmp_path)])

    result = runner.invoke(app, ["validate", str(tmp_path)])
    assert result.exit_code == 0
    assert "valid" in result.stdout.lower()

    result = runner.invoke(app, ["clear", str(tmp_path), "--yes"])
    assert result.exit_code == 0
    assert not tmp_path.exists()


def test_cli_validate_mismatch(tmp_path):
    runner = CliRunner()
    runner.invoke(app, ["init", str(tmp_path)])

    meta_path = tmp_path / "meta.yaml"
    import yaml

    meta = yaml.safe_load(meta_path.read_text())
    meta["embedding_dim"] = 2
    meta_path.write_text(yaml.safe_dump(meta))

    result = runner.invoke(app, ["validate", str(tmp_path)])
    assert result.exit_code != 0


def test_cli_talk(tmp_path, monkeypatch):
    runner = CliRunner()
    runner.invoke(app, ["init", str(tmp_path)])
    runner.invoke(app, ["add", "--agent-name", str(tmp_path), "--text", "hello world"])

    prompts = {}

    class Dummy:
        def __init__(self, *a, **kw):
            pass

        tokenizer = staticmethod(lambda text, return_tensors=None: {"input_ids": [text.split()]})
        model = type("M", (), {"config": type("C", (), {"n_positions": 50})()})()
        max_new_tokens = 10

        def prepare_prompt(self, agent, prompt, **kw):
            return prompt

        def reply(self, prompt: str) -> str:
            prompts["text"] = prompt
            return "response"

    monkeypatch.setattr("gist_memory.local_llm.LocalChatModel", Dummy)
    result = runner.invoke(
        app, ["talk", "--agent-name", str(tmp_path), "--message", "hi?"]
    )
    assert result.exit_code == 0
    assert "response" in result.stdout
    assert "hello world" in prompts["text"]
    assert "User asked" in prompts["text"]


def test_talk_command_calls_receive_channel_message(tmp_path, monkeypatch):
    runner = CliRunner()
    tmp_path.mkdir(exist_ok=True)

    called = {}

    class DummyAgent:
        store = type("S", (), {"meta": {}})()

        def receive_channel_message(self, src, msg, mgr=None):
            called["src"] = src
            called["msg"] = msg
            called["mgr"] = mgr
            return {"reply": "ok"}

    monkeypatch.setattr("gist_memory.cli._load_agent", lambda p: DummyAgent())

    class DummyLLM:
        def __init__(self, *a, **kw):
            pass

        def prepare_prompt(self, agent, prompt, **kw):
            return prompt

        def reply(self, prompt):
            return "ok"

        model = type("M", (), {"config": type("C", (), {"n_positions": 50})()})()
        max_new_tokens = 10

    monkeypatch.setattr("gist_memory.local_llm.LocalChatModel", DummyLLM)

    result = runner.invoke(
        app, ["talk", "--agent-name", str(tmp_path), "--message", "hi?"]
    )
    assert result.exit_code == 0
    assert called["src"] == "cli"
    assert called["msg"] == "hi?"
    assert called["mgr"] is not None





def test_cli_download_chat_model(monkeypatch):
    runner = CliRunner()

    calls = []

    class Dummy:
        @staticmethod
        def from_pretrained(name, **kw):
            calls.append(name)
            return Dummy()

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", Dummy.from_pretrained
    )
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained", Dummy.from_pretrained
    )

    result = runner.invoke(app, ["download-chat-model", "--model-name", "foo"])
    assert result.exit_code == 0
    assert "foo" in calls


def test_cli_logging(tmp_path):
    log_path = tmp_path / "cli.log"
    runner = CliRunner()
    runner.invoke(app, ["init", str(tmp_path)])
    runner.invoke(app, ["add", "--agent-name", str(tmp_path), "--text", "alpha"])
    result = runner.invoke(
        app,
        [
            "--log-file",
            str(log_path),
            "--verbose",
            "query",
            "--agent-name",
            str(tmp_path),
            "--query-text",
            "alpha",
        ],
    )
    assert result.exit_code == 0
    assert log_path.exists()
    assert log_path.read_text() != ""


def test_cli_corrupt_store(tmp_path):
    runner = CliRunner()
    runner.invoke(app, ["init", str(tmp_path)])
    meta_path = tmp_path / "meta.yaml"
    import yaml

    meta = yaml.safe_load(meta_path.read_text())
    meta["embedding_dim"] = 2
    meta_path.write_text(yaml.safe_dump(meta))

    result = runner.invoke(
        app,
        ["query", "--agent-name", str(tmp_path), "--query-text", "hi"],
    )
    assert result.exit_code != 0
    assert "Brain data is corrupted" in result.stderr


