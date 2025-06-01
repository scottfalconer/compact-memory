def test_experiment_optimize_runs(tmp_path):
    script = tmp_path / "opt.py"
    script.write_text("print('ok')")

    from gist_memory.cli import app
    from typer.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(app, ["experiment", "optimize", str(script)])
    assert result.exit_code == 0
    assert "ok" in result.stdout
