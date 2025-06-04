from pathlib import Path
import json
from typer.testing import CliRunner
from compact_memory.cli import app
from compact_memory.compression import (
    CompressionStrategy,
    CompressedMemory,
    CompressionTrace,
    register_compression_strategy,
)


class DummyTruncStrategy(CompressionStrategy):
    id = "dummy_trunc"

    def compress(self, text_or_chunks, llm_token_budget, **kwargs):
        text = (
            text_or_chunks
            if isinstance(text_or_chunks, str)
            else " ".join(text_or_chunks)
        )
        truncated = text[:llm_token_budget]
        return CompressedMemory(text=truncated), CompressionTrace(
            strategy_name=self.id,
            strategy_params={"llm_token_budget": llm_token_budget},
            input_summary={"input_length": len(text)},
            steps=[{"type": "truncate"}],
            output_summary={"final_length": len(truncated)},
            final_compressed_object_preview=truncated,
        )


register_compression_strategy(DummyTruncStrategy.id, DummyTruncStrategy)


def _env(tmp_path: Path) -> dict[str, str]:
    return {"COMPACT_MEMORY_COMPACT_MEMORY_PATH": str(tmp_path)}


runner = CliRunner()


def test_compress_text_option(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "compress",
            "--text",
            "hello world",
            "--strategy",
            "none",
            "--budget",
            "10",
        ],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0
    assert "hello" in result.stdout


def test_compress_stdin(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "compress",
            "--text",
            "-",
            "--strategy",
            "none",
            "--budget",
            "10",
        ],
        input="stdin text",
        env=_env(tmp_path),
    )
    assert result.exit_code == 0
    assert "stdin" in result.stdout


def test_compress_file(tmp_path: Path):
    file_path = tmp_path / "inp.txt"
    file_path.write_text("file text")
    result = runner.invoke(
        app,
        [
            "compress",
            "--file",
            str(file_path),
            "--strategy",
            "none",
            "--budget",
            "10",
        ],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0
    assert "file" in result.stdout


def test_compress_directory_recursive(tmp_path: Path):
    dir_path = tmp_path / "data"
    dir_path.mkdir()
    (dir_path / "a.txt").write_text("aaa")
    sub = dir_path / "sub"
    sub.mkdir()
    (sub / "b.txt").write_text("bbb")
    out_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "compress",
            "--dir",
            str(dir_path),
            "--strategy",
            "none",
            "--budget",
            "5",
            "--recursive",
            "--output-dir",
            str(out_dir),
        ],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0
    assert (out_dir / "a.txt").exists()
    assert (out_dir / "sub" / "b.txt").exists()


def test_compress_invalid_combo(tmp_path: Path):
    file_path = tmp_path / "foo.txt"
    file_path.write_text("foo")
    result = runner.invoke(
        app,
        [
            "compress",
            "--file",
            str(file_path),
            "--text",
            "oops",
            "--strategy",
            "none",
            "--budget",
            "5",
        ],
        env=_env(tmp_path),
    )
    assert result.exit_code != 0
    assert "specify exactly ONE" in result.stderr


def test_compress_file_output_file(tmp_path: Path):
    file_path = tmp_path / "input.txt"
    file_path.write_text("sample text")
    out_path = tmp_path / "out.txt"
    result = runner.invoke(
        app,
        [
            "compress",
            "--file",
            str(file_path),
            "--strategy",
            "none",
            "--budget",
            "100",
            "-o",
            str(out_path),
        ],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0
    assert out_path.read_text() == "sample text"


def test_compress_file_invalid_recursive(tmp_path: Path):
    file_path = tmp_path / "file.txt"
    file_path.write_text("text")
    result = runner.invoke(
        app,
        [
            "compress",
            "--file",
            str(file_path),
            "--strategy",
            "none",
            "--budget",
            "5",
            "--recursive",
        ],
        env=_env(tmp_path),
    )
    assert result.exit_code != 0
    assert "--recursive is only valid with --dir" in result.stderr


def test_compress_dir_pattern_and_output(tmp_path: Path):
    dir_path = tmp_path / "data"
    dir_path.mkdir()
    (dir_path / "a.txt").write_text("a")
    (dir_path / "b.md").write_text("b")
    sub = dir_path / "sub"
    sub.mkdir()
    (sub / "c.txt").write_text("c")
    out_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "compress",
            "--dir",
            str(dir_path),
            "--strategy",
            "none",
            "--budget",
            "10",
            "--recursive",
            "--pattern",
            "*.txt",
            "--output-dir",
            str(out_dir),
        ],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0
    assert (out_dir / "a.txt").exists()
    assert (out_dir / "sub" / "c.txt").exists()
    assert not (out_dir / "b.md").exists()


def test_compress_invalid_strategy(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "compress",
            "--text",
            "hello",
            "--strategy",
            "bogus",
            "--budget",
            "5",
        ],
        env=_env(tmp_path),
    )
    assert result.exit_code != 0
    assert "Unknown compression strategy" in result.stderr


def test_compress_budget_truncation(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "compress",
            "--text",
            "abcdef",
            "--strategy",
            DummyTruncStrategy.id,
            "--budget",
            "2",
        ],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0
    assert result.stdout.strip() == "ab"


def test_compress_output_trace(tmp_path: Path):
    trace_path = tmp_path / "trace.json"
    result = runner.invoke(
        app,
        [
            "compress",
            "--text",
            "hi there",
            "--strategy",
            "none",
            "--budget",
            "10",
            "--output-trace",
            str(trace_path),
        ],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0
    data = json.loads(trace_path.read_text())
    assert data["strategy_name"] == "none"
    for key in ["strategy_params", "input_summary", "output_summary", "steps"]:
        assert key in data
    assert data["output_summary"]["output_length"] == len("hi there")


def test_compress_output_trace_details(tmp_path: Path):
    """Trace file should contain step metadata for strategy."""
    trace_path = tmp_path / "trace.json"
    result = runner.invoke(
        app,
        [
            "compress",
            "--text",
            "hello world",
            "--strategy",
            DummyTruncStrategy.id,
            "--budget",
            "3",
            "--output-trace",
            str(trace_path),
        ],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0
    data = json.loads(trace_path.read_text())
    assert data["strategy_name"] == DummyTruncStrategy.id
    assert data["steps"] == [{"type": "truncate"}]
    assert data["output_summary"]["final_length"] == len("hello world"[:3])


def test_compress_verbose_stats(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "compress",
            "--text",
            "hello world",
            "--strategy",
            "none",
            "--budget",
            "20",
            "--verbose-stats",
        ],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0
    assert "Original tokens" in result.stdout
    assert "Compressed tokens" in result.stdout


def test_compress_nonexistent_file(tmp_path: Path):
    file_path = tmp_path / "missing.txt"
    result = runner.invoke(
        app,
        [
            "compress",
            "--file",
            str(file_path),
            "--strategy",
            "none",
            "--budget",
            "5",
        ],
        env=_env(tmp_path),
    )
    assert result.exit_code != 0
    assert "Invalid value for '--file'" in result.stderr


def test_compress_nonexistent_dir(tmp_path: Path):
    dir_path = tmp_path / "nope"
    result = runner.invoke(
        app,
        [
            "compress",
            "--dir",
            str(dir_path),
            "--strategy",
            "none",
            "--budget",
            "5",
        ],
        env=_env(tmp_path),
    )
    assert result.exit_code != 0
    assert "Invalid value for '--dir'" in result.stderr


def test_compress_uses_default_strategy(tmp_path: Path):
    env = _env(tmp_path)
    env["COMPACT_MEMORY_DEFAULT_STRATEGY_ID"] = "none"
    result = runner.invoke(
        app,
        ["compress", "--text", "foobar", "--budget", "10"],
        env=env,
    )
    assert result.exit_code == 0
    assert "foobar" in result.stdout


def test_compress_override_default_strategy(tmp_path: Path):
    env = _env(tmp_path)
    env["COMPACT_MEMORY_DEFAULT_STRATEGY_ID"] = DummyTruncStrategy.id
    result = runner.invoke(
        app,
        [
            "compress",
            "--text",
            "abcdef",
            "--strategy",
            "none",
            "--budget",
            "10",
        ],
        env=env,
    )
    assert result.exit_code == 0
    assert result.stdout.strip() == "abcdef"
