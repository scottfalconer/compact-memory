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


def test_cli_strategy_inspect(tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, ["init", "--memory-path", str(tmp_path), "--name", "tester"])
    assert result.exit_code == 0

    from gist_memory.utils import load_agent

    agent = load_agent(tmp_path)
    agent.add_memory("hello world")
    agent.store.save()

    result = runner.invoke(
        app,
        [
            "strategy",
            "inspect",
            "prototype",
            "--memory-path",
            str(tmp_path),
            "--list-prototypes",
        ],
    )
    assert result.exit_code == 0
    assert "hello world" in result.stdout


def test_cli_stats(tmp_path):
    runner = CliRunner()
    runner.invoke(app, ["init", "--memory-path", str(tmp_path), "--name", "tester"])
    from gist_memory.utils import load_agent

    agent = load_agent(tmp_path)
    agent.add_memory("alpha")
    agent.store.save()

    result = runner.invoke(app, ["stats", "--memory-path", str(tmp_path), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout.strip())
    assert data["prototypes"] == 1


def test_cli_validate_and_clear(tmp_path):
    runner = CliRunner()
    runner.invoke(app, ["init", "--memory-path", str(tmp_path)])

    result = runner.invoke(app, ["validate", "--memory-path", str(tmp_path)])
    assert result.exit_code == 0
    assert "valid" in result.stdout.lower()

    result = runner.invoke(app, ["clear", "--memory-path", str(tmp_path), "--yes"])
    assert result.exit_code == 0
    assert not tmp_path.exists()


def test_cli_validate_mismatch(tmp_path):
    runner = CliRunner()
    runner.invoke(app, ["init", "--memory-path", str(tmp_path)])

    meta_path = tmp_path / "meta.yaml"
    import yaml

    meta = yaml.safe_load(meta_path.read_text())
    meta["embedding_dim"] = 2
    meta_path.write_text(yaml.safe_dump(meta))

    result = runner.invoke(app, ["validate", "--memory-path", str(tmp_path)])
    assert result.exit_code != 0


def test_cli_talk(tmp_path, monkeypatch):
    runner = CliRunner()
    runner.invoke(app, ["init", "--memory-path", str(tmp_path)])
    from gist_memory.utils import load_agent

    agent = load_agent(tmp_path)
    agent.add_memory("hello world")
    agent.store.save()

    prompts = {}

    class Dummy:
        def __init__(self, *a, **kw):
            pass

        tokenizer = staticmethod(lambda text, return_tensors=None: {"input_ids": [text.split()]})
        model = type("M", (), {"config": type("C", (), {"n_positions": 50})()})()
        max_new_tokens = 10

        def load_model(self):
            pass

        def prepare_prompt(self, agent, prompt, **kw):
            return prompt

        def reply(self, prompt: str) -> str:
            prompts["text"] = prompt
            return "response"

    monkeypatch.setattr("gist_memory.local_llm.LocalChatModel", Dummy)
    result = runner.invoke(
        app, ["talk", "--memory-path", str(tmp_path), "--message", "hi?"]
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

        def load_model(self):
            pass

        model = type("M", (), {"config": type("C", (), {"n_positions": 50})()})()
        max_new_tokens = 10

    monkeypatch.setattr("gist_memory.local_llm.LocalChatModel", DummyLLM)

    result = runner.invoke(
        app, ["talk", "--memory-path", str(tmp_path), "--message", "hi?"]
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
    runner.invoke(app, ["init", "--memory-path", str(tmp_path)])
    from gist_memory.utils import load_agent

    agent = load_agent(tmp_path)
    agent.add_memory("alpha")
    agent.store.save()
    result = runner.invoke(
        app,
        [
            "--log-file",
            str(log_path),
            "--verbose",
            "stats",
            "--memory-path",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0
    assert log_path.exists()
    assert log_path.read_text() != ""


def test_cli_corrupt_store(tmp_path):
    runner = CliRunner()
    runner.invoke(app, ["init", "--memory-path", str(tmp_path)])
    meta_path = tmp_path / "meta.yaml"
    import yaml

    meta = yaml.safe_load(meta_path.read_text())
    meta["embedding_dim"] = 2
    meta_path.write_text(yaml.safe_dump(meta))

    result = runner.invoke(
        app,
        ["stats", "--memory-path", str(tmp_path)],
    )
    assert result.exit_code != 0
    assert "Brain data is corrupted" in result.stderr



def test_cli_trace_inspect(tmp_path):
    runner = CliRunner()
    trace_file = tmp_path / "trace.json"
    data = {
        "strategy_name": "dummy",
        "strategy_params": {},
        "input_summary": {},
        "steps": [{"type": "filter_item", "details": {"reason": "test"}}],
    }
    trace_file.write_text(json.dumps(data))
    result = runner.invoke(app, ["trace", "inspect", str(trace_file)])
    assert result.exit_code == 0
    assert "dummy" in result.stdout
    assert "filter_item" in result.stdout


def test_cli_compress_string_stdout():
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["compress", "alpha bravo charlie delta", "--strategy", "none", "--budget", "3"],
    )
    assert result.exit_code == 0
    assert "alpha bravo charlie" in result.stdout


def test_cli_compress_string_to_file(tmp_path):
    runner = CliRunner()
    out_file = tmp_path / "out.txt"
    result = runner.invoke(
        app,
        [
            "compress",
            "alpha bravo charlie delta",
            "--strategy",
            "none",
            "--budget",
            "3",
            "-o",
            str(out_file),
        ],
    )
    assert result.exit_code == 0
    assert out_file.read_text() == "alpha bravo charlie"


def test_cli_compress_file(tmp_path):
    runner = CliRunner()
    input_file = tmp_path / "in.txt"
    input_file.write_text("one two three four")
    out_file = tmp_path / "out.txt"
    result = runner.invoke(
        app,
        [
            "compress",
            str(input_file),
            "--strategy",
            "none",
            "--budget",
            "2",
            "-o",
            str(out_file),
        ],
    )
    assert result.exit_code == 0
    assert out_file.read_text() == "one two"


def test_cli_compress_directory(tmp_path):
    runner = CliRunner()
    dir_in = tmp_path / "inputs"
    dir_in.mkdir()
    (dir_in / "a.txt").write_text("alpha bravo charlie")
    (dir_in / "b.txt").write_text("delta echo foxtrot")
    result = runner.invoke(
        app,
        ["compress", str(dir_in), "--strategy", "none", "--budget", "2"],
    )
    assert result.exit_code == 0
    assert (dir_in / "a_compressed.txt").read_text() == "alpha bravo"
    assert (dir_in / "b_compressed.txt").read_text() == "delta echo"


def test_cli_compress_invalid_strategy():
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["compress", "text", "--strategy", "bogus", "--budget", "3"],
    )
    assert result.exit_code != 0
    assert "Unknown compression strategy" in result.stderr


def test_cli_output_trace(tmp_path):
    runner = CliRunner()
    trace_file = tmp_path / "trace.json"
    result = runner.invoke(
        app,
        [
            "compress",
            "hello world",
            "--strategy",
            "none",
            "--budget",
            "5",
            "--output-trace",
            str(trace_file),
        ],
    )
    assert result.exit_code == 0
    assert trace_file.exists()


def test_cli_strategy_and_metric_list():
    runner = CliRunner()
    res = runner.invoke(app, ["strategy", "list"])
    assert res.exit_code == 0
    assert "none" in res.stdout
    res = runner.invoke(app, ["metric", "list"])
    assert res.exit_code == 0
    assert "exact_match" in res.stdout


def test_cli_evaluate_compression():
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "evaluate-compression",
            "abcde",
            "abcd",
            "--metric",
            "compression_ratio",
            "--json",
        ],
    )
    assert res.exit_code == 0
    data = json.loads(res.stdout.strip())
    assert "compression_ratio" in data


def test_cli_llm_prompt(monkeypatch, tmp_path):
    runner = CliRunner()

    class DummyProvider:
        def generate_response(self, prompt, model_name, max_new_tokens, **kw):
            DummyProvider.prompt = prompt
            return "ok"

        def get_token_budget(self, model_name, **kw):
            return 100

        def count_tokens(self, text, model_name, **kw):
            return len(text.split())

    monkeypatch.setattr(
        "gist_memory.llm_providers.OpenAIProvider",
        lambda: DummyProvider(),
    )
    monkeypatch.setattr(
        "gist_memory.llm_providers.GeminiProvider",
        lambda: DummyProvider(),
    )
    monkeypatch.setattr(
        "gist_memory.llm_providers.LocalTransformersProvider",
        lambda: DummyProvider(),
    )

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("modelx:\n  provider: openai\n  model_name: modelx")
    res = runner.invoke(
        app,
        [
            "llm-prompt",
            "--context",
            "foo",
            "--query",
            "bar",
            "--model",
            "modelx",
            "--llm-config",
            str(cfg),
        ],
    )
    assert res.exit_code == 0
    assert "ok" in res.stdout
    assert "foo" in DummyProvider.prompt
    assert "bar" in DummyProvider.prompt


def test_cli_evaluate_llm_response():
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "evaluate-llm-response",
            "foo",
            "bar",
            "--metric",
            "exact_match",
            "--json",
        ],
    )
    assert res.exit_code == 0
    data = json.loads(res.stdout.strip())
    assert "exact_match" in data
