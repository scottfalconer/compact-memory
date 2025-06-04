from pathlib import Path
from typer.testing import CliRunner
from compact_memory.cli import app


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
