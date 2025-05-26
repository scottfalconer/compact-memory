import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from click.testing import CliRunner
from gist_memory.cli import cli


def test_cli_ingest_and_query(tmp_path):
    runner = CliRunner()
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = runner.invoke(cli, ["--db-dir", str(tmp_path), "ingest", "hello world"])
        assert result.exit_code == 0
        result = runner.invoke(cli, ["--db-dir", str(tmp_path), "query", "hello", "--top", "1"])
        assert result.exit_code == 0
    finally:
        os.chdir(cwd)
