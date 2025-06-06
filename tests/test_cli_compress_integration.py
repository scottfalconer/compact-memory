import pytest
from typer.testing import CliRunner
from compact_memory.cli import app # import the Typer app
from pathlib import Path # Make sure Path is imported

runner = CliRunner()

def test_compress_text_input_stdout():
    result = runner.invoke(app, [
        "compress",
        "--text", "Sample input text to compress",
        "--engine", "none",
        "--budget", "200"
    ])
    assert result.exit_code == 0
    # With the "none" engine, output text should equal input (within budget limits)
    output = result.stdout.strip()
    assert output.startswith("Sample input text") # original content present

@pytest.mark.parametrize("engine_id", ["none", "prototype", "first_last"])
def test_compress_text_input_all_engines(engine_id):
    text = "This is a sample text that should be compressed using various engines to test their basic functionality."
    # For prototype engine, a memory path might be implicitly needed or created.
    # Let's provide one via env var to be safe, as some engines might try to load/save.
    # We'll use a dummy path that doesn't need to exist for "none" or "first_last"
    # but might be used by "prototype" if it tries to initialize a default store.
    # The CLI's main() function has logic for memory_path resolution.
    # Providing a default config or env var helps avoid interactive prompts.
    env_vars = {"COMPACT_MEMORY_PATH": "./dummy_cli_test_path"}

    # For prototype engine, we might also need a default model ID if it tries to load one.
    if engine_id == "prototype":
        env_vars["COMPACT_MEMORY_DEFAULT_MODEL_ID"] = "mock-model" # Assuming mock-model is handled or doesn't break things

    result = runner.invoke(app, [
        "compress", "--text", text,
        "--engine", engine_id, "--budget", "200"
    ], env=env_vars) # Pass env_vars here

    assert result.exit_code == 0, f"Engine {engine_id} failed with exit code {result.exit_code}\nStdout: {result.stdout}\nStderr: {result.stderr}"
    output = result.stdout.strip()
    assert output, f"Engine {engine_id} produced no output."

    if engine_id == "none":
        # NoCompressionEngine should return input truncated to budget
        assert output.startswith("This is a sample text")
    else:
        # Other engines should generally produce shorter output than input or different
        # For prototype, it might be significantly different due to its summarization nature.
        # For first_last, it will be shorter if text is long enough.
        # A simple check is that it's not identical or is shorter.
        # Given the budget of 200 and the text length, it's likely to be shorter or different.
        # If text is very short, compressed might be same or even slightly longer due to formatting/metadata for some engines.
        # The provided text is long enough that changes should occur.
        if len(text) > 50: # Only assert length change for reasonably long text
             assert len(output) < len(text) or output != text, f"Engine {engine_id} output was not different or shorter than input."
        else:
            assert output != text or engine_id == "first_last", f"Engine {engine_id} output was not different for short text."

def test_compress_file_input_stdout(tmp_path):
    # Setup: create a temporary file with some content
    input_file = tmp_path / "example.txt"
    input_file.write_text("Content to compress from a file.")

    result = runner.invoke(app, [
        "compress",
        "--file", str(input_file),
        "--engine", "none",
        "--budget", "200"
    ])
    assert result.exit_code == 0, f"CLI call failed: {result.stderr}"
    # The output should be printed to stdout (no --output specified)
    output = result.stdout.strip()
    assert "Content to compress from a file." in output

def test_compress_file_to_output_file(tmp_path):
    input_file = tmp_path / "input.txt"
    original_content = "ABC " * 100  # create a larger content
    input_file.write_text(original_content)
    out_file = tmp_path / "output.txt"

    budget = 50 # Small budget to ensure truncation by "none" engine

    result = runner.invoke(app, [
        "compress",
        "--file", str(input_file),
        "--engine", "none", # Using "none" engine for predictable truncation
        "--budget", str(budget), # Pass budget as string, Typer handles conversion
        "--output", str(out_file)
    ])
    assert result.exit_code == 0, f"CLI call failed: {result.stderr}"

    # The CLI should save output to the specified file
    assert out_file.exists(), "Output file was not created."

    # Verify content of the output file
    # "none" engine with a budget will truncate.
    # The exact tokenization and truncation logic for "none" engine needs to be considered.
    # Assuming it truncates based on character count if no tokenizer is found,
    # or a simple split if a basic tokenizer is used by default.
    # For "none" engine, it effectively takes a slice of the input text.
    # The budget is in tokens. "none" engine's behavior w.r.t token budget:
    # PrototypeEngine.compress (which NoCompressionEngine inherits its compress from, indirectly)
    # has a simple split tokenizer if tiktoken is not available.
    # Let's assume for "none" engine, budget roughly corresponds to characters for simplicity in this test.
    # A more robust way would be to check if content is a prefix and its length is related to budget.

    # The CLI prints a success message to stdout when using --output
    assert "Saved compressed output to" in result.stdout, "Confirmation message not found in stdout."
    assert str(out_file) in result.stdout, "Output file path not mentioned in confirmation message."

    # Check that the output file content is the beginning of the input file,
    # and its length is constrained (though not strictly equal to budget due to tokenization details).
    output_content = out_file.read_text()
    assert original_content.startswith(output_content)
    # This assertion is tricky because budget is tokens, output is text.
    # For "none" engine, it should be the initial part of the text.
    # A loose check on length:
    assert len(output_content) > 0, "Output file is empty."
    # If budget is 50 tokens, output char length will be > 50 usually.
    # And it should be less than original if truncation happened.
    if len(original_content) > budget * 2: # Heuristic: if original is much larger than budget in chars
        assert len(output_content) < len(original_content), "Output content was not truncated."

def test_compress_directory_input_to_output_dir(tmp_path):
    # Set up a temp directory with two text files
    dir_path = tmp_path / "data"
    dir_path.mkdir()
    (dir_path / "a.txt").write_text("Content of file A.")
    (dir_path / "b.txt").write_text("Content of file B which is a bit longer.")

    # Target output directory for compressed files
    out_dir = tmp_path / "out"
    # Note: out_dir should not be created beforehand by the test,
    # the CLI command should create it if it doesn't exist.

    result = runner.invoke(app, [
        "compress",
        "--dir", str(dir_path),
        "--engine", "none",
        "--budget", "100", # Sufficient budget for "none" engine not to alter content much
        "--output-dir", str(out_dir)
    ])
    assert result.exit_code == 0, f"CLI call failed: {result.stderr}\n{result.stdout}"

    # The CLI should process both files and create outputs in out_dir
    compressed_a = out_dir / "a.txt" # Original filenames are preserved in output_dir
    compressed_b = out_dir / "b.txt"

    assert compressed_a.exists(), "Compressed file for a.txt not found in output directory."
    assert compressed_b.exists(), "Compressed file for b.txt not found in output directory."

    # Verify content (with "none" engine, content should be same as original)
    assert compressed_a.read_text() == "Content of file A."
    assert compressed_b.read_text() == "Content of file B which is a bit longer."

    # Stdout should show processing messages for each file
    stdout = result.stdout
    assert f"Processing {dir_path / 'a.txt'}" in stdout or f"Processing {Path('data/a.txt')}" in stdout # Path representation might vary
    assert f"Processing {dir_path / 'b.txt'}" in stdout or f"Processing {Path('data/b.txt')}" in stdout

    # Check for "Saved compressed output to..." messages for each file
    assert f"Saved compressed output to {compressed_a}" in stdout
    assert f"Saved compressed output to {compressed_b}" in stdout

def test_compress_directory_input_default_output(tmp_path):
    dir_path = tmp_path / "data_default"
    dir_path.mkdir()
    file_a = dir_path / "file_one.txt"
    file_b = dir_path / "file_two.txt"
    file_a.write_text("Default output for file one.")
    file_b.write_text("Default output for file two.")

    result = runner.invoke(app, [
        "compress",
        "--dir", str(dir_path),
        "--engine", "none",
        "--budget", "150"
    ])
    assert result.exit_code == 0, f"CLI call failed: {result.stderr}\n{result.stdout}"

    # Verify that _compressed.txt files are created alongside originals
    compressed_a_expected = dir_path / "file_one_compressed.txt"
    compressed_b_expected = dir_path / "file_two_compressed.txt"

    assert compressed_a_expected.exists(), "file_one_compressed.txt not found."
    assert compressed_b_expected.exists(), "file_two_compressed.txt not found."

    assert compressed_a_expected.read_text() == "Default output for file one."
    assert compressed_b_expected.read_text() == "Default output for file two."

    # Stdout should show processing messages and save confirmations
    stdout = result.stdout
    assert f"Processing {file_a}" in stdout
    assert f"Processing {file_b}" in stdout
    assert f"Saved compressed output to {compressed_a_expected}" in stdout
    assert f"Saved compressed output to {compressed_b_expected}" in stdout

def test_compress_empty_directory(tmp_path):
    empty_dir = tmp_path / "empty_data"
    empty_dir.mkdir()

    result = runner.invoke(app, [
        "compress",
        "--dir", str(empty_dir),
        "--engine", "none",
        "--budget", "100"
    ])
    # According to the issue doc, this should be exit_code 0
    assert result.exit_code == 0, f"CLI call failed for empty dir: {result.stderr}\n{result.stdout}"
    assert "No matching files found." in result.stdout

def test_compress_directory_no_matching_files(tmp_path):
    data_dir = tmp_path / "mismatched_files"
    data_dir.mkdir()
    (data_dir / "document.md").write_text("This is a markdown file.")
    (data_dir / "image.jpg").write_text("This is not text.") # Content doesn't matter

    result = runner.invoke(app, [
        "compress",
        "--dir", str(data_dir),
        "--engine", "none",
        "--budget", "100",
        # Using default pattern "*.txt"
    ])
    assert result.exit_code == 0, f"CLI call failed for dir with no .txt files: {result.stderr}\n{result.stdout}"
    assert "No matching files found." in result.stdout

def test_compress_unknown_engine():
    result = runner.invoke(app, [
        "compress", "--text", "test",
        "--engine", "does_not_exist_engine",
        "--budget", "100"
    ], mix_stderr=False) # Capture stderr separately
    assert result.exit_code != 0
    assert "Unknown compression engine" in result.stderr
    assert "does_not_exist_engine" in result.stderr

def test_compress_file_not_found(tmp_path):
    fake_file = tmp_path / "no_such_file.txt"  # do not create it
    result = runner.invoke(app, [
        "compress",
        "--file", str(fake_file),
        "--engine", "none",
        "--budget", "50"
    ], mix_stderr=False)
    assert result.exit_code != 0
    # Typer/Click's built-in File type validation might produce this error message
    # before my custom "File not found:" message in the command's logic.
    # The exact message depends on when Typer performs its `exists=True` check.
    # "Invalid value for '--file': File '/tmp/pytest-of-user/pytest-0/test_compress_file_not_found0/no_such_file.txt' does not exist."
    assert "Invalid value for '--file'" in result.stderr
    assert str(fake_file) in result.stderr
    assert "does not exist" in result.stderr

def test_compress_directory_not_found(tmp_path):
    fake_dir = tmp_path / "no_such_dir" # do not create it
    result = runner.invoke(app, [
        "compress",
        "--dir", str(fake_dir),
        "--engine", "none",
        "--budget", "50"
    ], mix_stderr=False)
    assert result.exit_code != 0
    # Similar to file, Typer's `exists=True` for Path will likely trigger this.
    assert "Invalid value for '--dir'" in result.stderr
    assert str(fake_dir) in result.stderr
    assert "does not exist" in result.stderr

def test_compress_no_input_provided():
    result = runner.invoke(app, [
        "compress", # No --text, --file, or --dir
        "--engine", "none",
        "--budget", "100"
    ], mix_stderr=False)
    assert result.exit_code != 0
    assert "specify exactly ONE of --text / --file / --dir" in result.stderr.strip()

@pytest.mark.parametrize("input_args", [
    # Providing dummy paths that don't need to exist, as the error should be caught before path access.
    # However, if Typer's `exists=True` runs first for --file or --dir, it might fail there.
    # For this test to robustly check the "exactly ONE" logic, we assume that check comes before
    # or is independent of the `exists=True` validation for individual path options.
    # If `exists=True` always runs first, these tests might report "file not found" instead.
    # The CLI code shows the check `sum(x is not None for x in (text, file, dir)) != 1`
    # happens before file system access for the content of the files/dirs.
    # So, providing non-existent paths here should be fine for testing this specific error.
    ["--text", "hi", "--file", "dummy_non_existent_file.txt"],
    ["--text", "hi", "--dir", "dummy_non_existent_dir"],
    ["--file", "dummy_non_existent_file.txt", "--dir", "dummy_non_existent_dir"],
    ["--text", "hi", "--file", "dummy.txt", "--dir", "dummy_dir"] # All three
])
def test_compress_multiple_inputs_error(tmp_path, input_args):
    # Create dummy files/dirs if Typer's validation requires them to exist,
    # even if our target validation (mutual exclusivity) happens earlier.
    # This makes the test more robust against Typer's path validation order.
    if "--file" in input_args:
        idx = input_args.index("--file")
        f_path = tmp_path / input_args[idx+1]
        f_path.touch() # Create the dummy file
        input_args[idx+1] = str(f_path) # Update arg to actual path

    if "--dir" in input_args:
        idx = input_args.index("--dir")
        d_path = tmp_path / input_args[idx+1]
        d_path.mkdir(exist_ok=True) # Create the dummy dir
        input_args[idx+1] = str(d_path) # Update arg to actual path

    command = ["compress", "--engine", "none", "--budget", "50"] + input_args

    result = runner.invoke(app, command, mix_stderr=False)

    assert result.exit_code != 0, f"CLI call with {input_args} did not fail as expected. Stderr: {result.stderr}"
    assert "specify exactly ONE of --text / --file / --dir" in result.stderr.strip()
