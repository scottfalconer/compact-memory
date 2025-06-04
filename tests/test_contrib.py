from typer.testing import CliRunner

from contrib import enable_all_contrib_strategies
from compact_memory.cli import app
from compact_memory.compression import (
    get_compression_strategy,
    available_strategies,
)


def test_enable_all_registers_contrib():
    enable_all_contrib_strategies()
    assert "chained" in available_strategies()
    cls = get_compression_strategy("chained")
    assert cls.__name__ == "ChainedStrategy"


def test_cli_list_strategies_include_contrib():
    runner = CliRunner()
    res = runner.invoke(app, ["dev", "list-strategies"])
    assert "chained" not in res.stdout
    res = runner.invoke(app, ["dev", "list-strategies", "--include-contrib"])
    assert res.exit_code == 0
    assert "chained" in res.stdout
