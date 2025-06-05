from pathlib import Path
import json
from typer.testing import CliRunner
from compact_memory.cli import app
from compact_memory.engine_registry import register_compression_engine
from compact_memory.engines import (
    BaseCompressionEngine,
    CompressedMemory,
    CompressionTrace,
    PrototypeEngine, # Added
    load_engine      # Added
)


class DummyTruncEngine(BaseCompressionEngine):
    id = "dummy_trunc"

    def compress(self, text_or_chunks, llm_token_budget, **kwargs):
        text = (
            text_or_chunks
            if isinstance(text_or_chunks, str)
            else " ".join(text_or_chunks)
        )
        truncated = text[:llm_token_budget]
        return CompressedMemory(text=truncated), CompressionTrace(
            engine_name=self.id,
            strategy_params={"llm_token_budget": llm_token_budget},
            input_summary={"input_length": len(text)},
            steps=[{"type": "truncate"}],
            output_summary={"final_length": len(truncated)},
            final_compressed_object_preview=truncated,
        )


register_compression_engine(DummyTruncEngine.id, DummyTruncEngine)


def _env(tmp_path: Path) -> dict[str, str]:
    # Ensure this matches the expected env var name used by the application's config loader
    return {
        "COMPACT_MEMORY_PATH": str(tmp_path),
        "COMPACT_MEMORY_DEFAULT_ENGINE_ID": "none", # Add defaults used by some tests
        "COMPACT_MEMORY_DEFAULT_MODEL_ID": "tiny-gpt2", # Add defaults used by some tests
    }


runner = CliRunner()


def test_compress_text_option(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "compress",
            "--text",
            "hello world",
            "--engine",
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
            "--engine",
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
            "--engine",
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
            "--engine",
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
            "--engine",
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
            "--engine",
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
            "--engine",
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
            "--engine",
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
            "--engine",
            "bogus",
            "--budget",
            "5",
        ],
        env=_env(tmp_path),
    )
    assert result.exit_code != 0
    assert "Unknown compression engine" in result.stderr


def test_compress_budget_truncation(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "compress",
            "--text",
            "abcdef",
            "--engine",
            DummyTruncEngine.id,
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
            "--engine",
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
    assert data["engine_name"] == "none"
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
            "--engine",
            DummyTruncEngine.id,
            "--budget",
            "3",
            "--output-trace",
            str(trace_path),
        ],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0
    data = json.loads(trace_path.read_text())
    assert data["engine_name"] == DummyTruncEngine.id
    assert data["steps"] == [{"type": "truncate"}]
    assert data["output_summary"]["final_length"] == len("hello world"[:3])


def test_compress_verbose_stats(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "compress",
            "--text",
            "hello world",
            "--engine",
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
            "--engine",
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
            "--engine",
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
    env["COMPACT_MEMORY_DEFAULT_ENGINE_ID"] = "none"
    result = runner.invoke(
        app,
        ["compress", "--text", "foobar", "--budget", "10"],
        env=env,
    )
    assert result.exit_code == 0
    assert "foobar" in result.stdout


def test_compress_override_default_strategy(tmp_path: Path):
    env = _env(tmp_path)
    env["COMPACT_MEMORY_DEFAULT_ENGINE_ID"] = DummyTruncEngine.id
    result = runner.invoke(
        app,
        [
            "compress",
            "--text",
            "abcdef",
            "--engine",
            "none",
            "--budget",
            "10",
        ],
        env=env,
    )
    assert result.exit_code == 0
    assert result.stdout.strip() == "abcdef"


def test_compress_to_memory_with_prototype_engine(tmp_path: Path, patch_embedding_model):
    prototype_store_path = tmp_path / "proto_store_mem_1"

    # Initialize a prototype engine store
    init_result = runner.invoke(
        app,
        ["engine", "init", "--engine", "prototype", str(prototype_store_path)],
        env=_env(tmp_path),
    )
    assert init_result.exit_code == 0, f"Engine init failed: {init_result.stderr}"

    raw_text = "This is a test sentence for prototype engine memory, it should be found."

    # Compress (using "none" engine for one-shot) and ingest into the prototype_store_path
    compress_result = runner.invoke(
        app,
        [
            "compress",
            "--memory-path",
            str(prototype_store_path),
            "--text",
            raw_text,
            "--engine", # This is the one-shot compression engine
            "none",
            "--budget",
            "1000", # Large budget, so raw_text is unchanged by "none" compressor
        ],
        env=_env(tmp_path),
    )
    assert compress_result.exit_code == 0, f"Compress to memory failed: {compress_result.stderr}"

    # Programmatically load the engine to verify ingestion
    loaded_engine = load_engine(prototype_store_path)
    assert isinstance(loaded_engine, PrototypeEngine), "Loaded engine is not a PrototypeEngine"

    query_results = loaded_engine.query("test sentence for prototype")
    assert query_results.get("memories"), "Query returned no memories"
    found = False
    for mem in query_results.get("memories", []):
        if raw_text in mem.get("text", ""):
            found = True
            break
    assert found, f"Original text '{raw_text}' not found in recalled memories."


def test_compress_to_memory_one_shot_trunc_then_prototype(tmp_path: Path, patch_embedding_model):
    prototype_store_path_2 = tmp_path / "proto_store_mem_2"

    # Initialize another prototype engine store
    init_result_2 = runner.invoke(
        app,
        ["engine", "init", "--engine", "prototype", str(prototype_store_path_2)],
        env=_env(tmp_path),
    )
    assert init_result_2.exit_code == 0, f"Engine init failed: {init_result_2.stderr}"

    long_text = "This is a very long sentence that is intended to be truncated by the dummy_trunc engine before being ingested into the prototype store."
    trunc_budget = 20 # Small budget for truncation
    expected_ingested_text = long_text[:trunc_budget]

    # Compress (using DummyTruncEngine for one-shot) and ingest into the prototype_store_path_2
    compress_result_2 = runner.invoke(
        app,
        [
            "compress",
            "--memory-path",
            str(prototype_store_path_2),
            "--text",
            long_text,
            "--engine", # This is the one-shot compression engine
            DummyTruncEngine.id,
            "--budget",
            str(trunc_budget),
        ],
        env=_env(tmp_path),
    )
    assert compress_result_2.exit_code == 0, f"Compress to memory failed: {compress_result_2.stderr}"

    # Programmatically load the engine
    loaded_engine_2 = load_engine(prototype_store_path_2)
    assert isinstance(loaded_engine_2, PrototypeEngine), "Loaded engine is not a PrototypeEngine"

    # Query for the truncated part (should be found)
    query_results_trunc = loaded_engine_2.query(expected_ingested_text)
    assert query_results_trunc.get("memories"), f"Query for truncated text '{expected_ingested_text}' returned no memories"
    found_truncated = False
    for mem in query_results_trunc.get("memories", []):
        if expected_ingested_text in mem.get("text", ""):
            found_truncated = True
            break
    assert found_truncated, f"Expected ingested text '{expected_ingested_text}' not found."

    # Query for the part that should have been truncated away (should not be found)
    # Making the query more specific to the part that should be gone.
    # If the query is too broad, it might match the truncated part.
    truncated_away_part = long_text[trunc_budget + 5 : trunc_budget + 15] # A slice of the part that's gone
    if truncated_away_part: # Ensure there's a non-empty string to query for
        query_results_full = loaded_engine_2.query(truncated_away_part)
        if query_results_full.get("memories"):
            found_non_truncated = False
            for mem in query_results_full.get("memories", []):
                # Check if the text *only* contains the truncated part, not the full long_text
                if expected_ingested_text in mem.get("text", "") and long_text[trunc_budget:] not in mem.get("text", "") :
                    pass # This is expected
                elif long_text[trunc_budget:] in mem.get("text", ""):
                    found_non_truncated = True
                    break
            assert not found_non_truncated, f"Query for text that should have been truncated away ('{truncated_away_part}') unexpectedly found matches containing the non-truncated part."
        else:
            assert not query_results_full.get("memories"), f"Query for text that should have been truncated ('{truncated_away_part}') returned memories when it should not have."
    else:
        # If truncated_away_part is empty, this part of the test is skipped.
        pass
