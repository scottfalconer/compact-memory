import pytest
from typer.testing import CliRunner
from compact_memory.cli import app  # import the Typer app
from pathlib import Path  # Make sure Path is imported
from tests.test_cli_compress import DummyTruncEngine # Try importing now that tests is a package
from compact_memory.engines.registry import register_compression_engine, available_engines

# Ensure DummyTruncEngine is registered for these integration tests
if DummyTruncEngine.id not in available_engines():
    register_compression_engine(DummyTruncEngine.id, DummyTruncEngine)

runner = CliRunner(mix_stderr=False)


def test_compress_text_input_stdout():
    result = runner.invoke(
        app,
        [
            "compress",
            "--text",
            "Sample input text to compress",
            "--engine",
            "none",
            "--budget",
            "200",
        ],
    )
    assert result.exit_code == 0
    # With the "none" engine, output text should equal input (within budget limits)
    output = result.stdout.strip()
    assert output.startswith("Sample input text")  # original content present


@pytest.mark.parametrize("engine_id", ["none", "first_last", "stopword_pruner"])
def test_compress_text_input_all_engines(engine_id):
    text = "This is a sample text that should be compressed using various engines to test their basic functionality."
    # Provide a dummy path via env var so engines that rely on a memory path have one.
    # The CLI's main() function has logic for memory_path resolution.
    # Providing a default config or env var helps avoid interactive prompts.
    env_vars = {"COMPACT_MEMORY_PATH": "./dummy_cli_test_path"}

    result = runner.invoke(
        app,
        ["compress", "--text", text, "--engine", engine_id, "--budget", "200"],
        env=env_vars,
    )  # Pass env_vars here

    assert (
        result.exit_code == 0
    ), f"Engine {engine_id} failed with exit code {result.exit_code}\nStdout: {result.stdout}\nStderr: {result.stderr}"
    output = result.stdout.strip()
    assert output, f"Engine {engine_id} produced no output."

    if engine_id == "none":
        # NoCompressionEngine should return input truncated to budget
        assert output.startswith("This is a sample text")
    else:
        # Other engines should generally produce shorter output than input or different.
        if len(text) > 50:
            assert (
                len(output) < len(text) or output != text
            ), f"Engine {engine_id} output was not different or shorter than input."
        else:
            assert (
                output != text or engine_id == "first_last"
            ), f"Engine {engine_id} output was not different for short text."


def test_compress_file_input_stdout(tmp_path):
    # Setup: create a temporary file with some content
    input_file = tmp_path / "example.txt"
    input_file.write_text("Content to compress from a file.")

    result = runner.invoke(
        app,
        ["compress", "--file", str(input_file), "--engine", "none", "--budget", "200"],
    )
    assert result.exit_code == 0, f"CLI call failed: {result.stderr}"
    # The output should be printed to stdout (no --output specified)
    output = result.stdout.strip()
    assert "Content to compress from a file." in output


def test_compress_file_to_output_file(tmp_path):
    input_file = tmp_path / "input.txt"
    original_content = "ABC " * 100  # create a larger content
    input_file.write_text(original_content)
    out_file = tmp_path / "output.txt"

    budget = 50  # Small budget to ensure truncation by "none" engine

    result = runner.invoke(
        app,
        [
            "compress",
            "--file",
            str(input_file),
            "--engine",
            "none",  # Using "none" engine for predictable truncation
            "--budget",
            str(budget),  # Pass budget as string, Typer handles conversion
            "--output",
            str(out_file),
        ],
    )
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
    assert (
        "Saved compressed output to" in result.stdout
    ), "Confirmation message not found in stdout."
    assert (
        str(out_file) in result.stdout
    ), "Output file path not mentioned in confirmation message."

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
    if (
        len(original_content) > budget * 2
    ):  # Heuristic: if original is much larger than budget in chars
        assert len(output_content) < len(
            original_content
        ), "Output content was not truncated."


# def test_compress_directory_input_to_output_dir(tmp_path):
    # # Set up a temp directory with two text files
    # # dir_path = tmp_path / "data" # This was the original line
    # # dir_path.mkdir() # This line was causing NameError, it belongs to the commented test
    # # (dir_path / "a.txt").write_text("Content of file A.")
    # # (dir_path / "b.txt").write_text("Content of file B which is a bit longer.")
    #
    # # Target output directory for compressed files
    # out_dir = tmp_path / "out"
    # # Note: out_dir should not be created beforehand by the test,
    # # the CLI command should create it if it doesn't exist.
    #
    # result = runner.invoke(
    #     app,
    #     [
    #         "compress",
    #         "--dir",
    #         str(dir_path),
    #         "--engine",
    #         "none",
    #         "--budget",
    #         "100",  # Sufficient budget for "none" engine not to alter content much
    #         "--output-dir",
    #         str(out_dir),
    #     ],
    # )
    # assert result.exit_code == 0, f"CLI call failed: {result.stderr}\n{result.stdout}"
    #
    # # The CLI should process both files and create outputs in out_dir
    # compressed_a = out_dir / "a.txt"  # Original filenames are preserved in output_dir
    # compressed_b = out_dir / "b.txt"
    #
    # assert (
    #     compressed_a.exists()
    # ), "Compressed file for a.txt not found in output directory."
    # assert (
    #     compressed_b.exists()
    # ), "Compressed file for b.txt not found in output directory."
    #
    # # Verify content (with "none" engine, content should be same as original)
    # assert compressed_a.read_text() == "Content of file A."
    # assert compressed_b.read_text() == "Content of file B which is a bit longer."
    #
    # # Stdout should show processing messages for each file
    # stdout = result.stdout
    # # assert (
    # #     f"Processing {dir_path / 'a.txt'}" in stdout
    # #     or f"Processing {Path('data/a.txt')}" in stdout
    # # )
    # # assert (
    # #     f"Processing {dir_path / 'b.txt'}" in stdout
    # #     or f"Processing {Path('data/b.txt')}" in stdout
    # # )
    # # # Check for "Saved compressed output to..." messages for each file
    # # # assert f"Saved compressed output to {compressed_a}" in stdout # These messages change with new logic
    # # # assert f"Saved compressed output to {compressed_b}" in stdout
    # # pass # Test is superseded by new directory compression logic


# def test_compress_directory_input_default_output(tmp_path):
    # dir_path = tmp_path / "data_default"
    # dir_path.mkdir()
    # file_a = dir_path / "file_one.txt"
    # file_b = dir_path / "file_two.txt"
    # file_a.write_text("Default output for file one.")
    # file_b.write_text("Default output for file two.")

    # result = runner.invoke(
    #     app, ["compress", "--dir", str(dir_path), "--engine", "none", "--budget", "150"]
    # )
    # assert result.exit_code == 0, f"CLI call failed: {result.stderr}\n{result.stdout}"

    # # Verify that _compressed.txt files are created alongside originals
    # compressed_a_expected = dir_path / "file_one_compressed.txt"
    # compressed_b_expected = dir_path / "file_two_compressed.txt"

    # assert compressed_a_expected.exists(), "file_one_compressed.txt not found."
    # assert compressed_b_expected.exists(), "file_two_compressed.txt not found."

    # assert compressed_a_expected.read_text() == "Default output for file one."
    # assert compressed_b_expected.read_text() == "Default output for file two."

    # # Stdout should show processing messages and save confirmations
    # stdout = result.stdout
    # assert f"Processing {file_a}" in stdout
    # assert f"Processing {file_b}" in stdout
    # assert f"Saved compressed output to {compressed_a_expected}" in stdout
    # assert f"Saved compressed output to {compressed_b_expected}" in stdout
    pass # Test is superseded by new directory compression logic


def test_compress_empty_directory(tmp_path):
    empty_dir = tmp_path / "empty_data"
    empty_dir.mkdir()

    result = runner.invoke(
        app,
        ["compress", "--dir", str(empty_dir), "--engine", "none", "--budget", "100"],
    )
    # According to the issue doc, this should be exit_code 0
    assert (
        result.exit_code == 0
    ), f"CLI call failed for empty dir: {result.stderr}\n{result.stdout}"
    assert "No files matching pattern" in result.stdout  # Made more specific


def test_compress_directory_no_matching_files(tmp_path):
    data_dir = tmp_path / "mismatched_files"
    data_dir.mkdir()
    (data_dir / "document.md").write_text("This is a markdown file.")
    (data_dir / "image.jpg").write_text("This is not text.")  # Content doesn't matter

    result = runner.invoke(
        app,
        [
            "compress",
            "--dir",
            str(data_dir),
            "--engine",
            "none",
            "--budget",
            "100",
            # Using default pattern "*.txt"
        ],
    )
    assert (
        result.exit_code == 0
    ), f"CLI call failed for dir with no .txt files: {result.stderr}\n{result.stdout}"
    assert "No files matching pattern" in result.stdout  # Made more specific


def test_compress_unknown_engine():
    result = runner.invoke(
        app,
        [
            "compress",
            "--text",
            "test",
            "--engine",
            "does_not_exist_engine",
            "--budget",
            "100",
        ],
    )  # Capture stderr separately
    assert result.exit_code != 0
    assert "Unknown one-shot compression engine" in result.stderr  # More specific
    assert "does_not_exist_engine" in result.stderr


def test_compress_file_not_found(tmp_path):
    fake_file = tmp_path / "no_such_file.txt"  # do not create it
    result = runner.invoke(
        app,
        ["compress", "--file", str(fake_file), "--engine", "none", "--budget", "50"],
    )
    assert result.exit_code != 0
    # Typer/Click's built-in File type validation might produce this error message
    # before my custom "File not found:" message in the command's logic.
    # The exact message depends on when Typer performs its `exists=True` check.
    # "Invalid value for '--file': File '/tmp/pytest-of-user/pytest-0/test_compress_file_not_found0/no_such_file.txt' does not exist."
    assert "Invalid value for '--file'" in result.stderr
    # assert str(fake_file) in result.stderr # This can be fragile with Rich formatting
    assert "does not exist" in result.stderr  # Typer's message includes this


def test_compress_directory_not_found(tmp_path):
    fake_dir = tmp_path / "no_such_dir"  # do not create it
    result = runner.invoke(
        app,
        ["compress", "--dir", str(fake_dir), "--engine", "none", "--budget", "50"],
    )
    assert result.exit_code != 0
    # Similar to file, Typer's `exists=True` for Path will likely trigger this.
    assert "Invalid value for '--dir'" in result.stderr
    assert str(fake_dir) in result.stderr
    assert "does not exist" in result.stderr


def test_compress_no_input_provided():
    result = runner.invoke(
        app,
        [
            "compress",  # No --text, --file, or --dir
            "--engine",
            "none",
            "--budget",
            "100",
        ],
    )
    assert result.exit_code != 0
    assert (
        "Specify exactly ONE of --text, --file, or --dir" in result.stderr.strip()
    )  # Adjusted to match actual


# --- New tests for consolidated directory compression ---

# def _expected_truncate_output(text: str, word_budget: int) -> str:
#     """
#     Helper to simulate simple word-based truncation for test predictions.
#     Note: The actual 'truncate' engine behavior depends on its tokenizer
#     and might differ slightly from this simple word split. This helper
#     is for creating a predictable *expected* output for tests.
#     """
#     words = text.split()
#     return " ".join(words[:word_budget])

def test_compress_directory_default_output_new(tmp_path):
    input_dir = tmp_path / "test_input_dir"
    input_dir.mkdir()
    (input_dir / "file1.txt").write_text("This is the first file content.")
    (input_dir / "file2.txt").write_text("This is the second file, it has some more words.")
    (input_dir / "another.md").write_text("This is a markdown file and should be ignored by default pattern.")

    # Expected combined content before compression (default pattern is *.txt)
    # Files are sorted by Path.glob, so file1.txt then file2.txt
    combined_content = "This is the first file content.\n\nThis is the second file, it has some more words."

    # word_budget = 10 # Example budget in words
    # Note: The 'truncate' engine uses its own tokenization.
    # This helper provides a simplified expectation.
    # expected_compressed_text = _expected_truncate_output(combined_content, word_budget) # This line caused NameError
    # Files are sorted by Path.glob, so file1.txt then file2.txt
    combined_content = "This is the first file content.\n\nThis is the second file, it has some more words."

    char_budget = 10 # Example budget in characters for dummy_trunc

    result = runner.invoke(
        app,
        [
            "compress",
            "--dir",
            str(input_dir),
            "--engine",
            "dummy_trunc", # Use "dummy_trunc" (engine ID directly)
            "--budget",
            str(char_budget),
        ],
    )

    assert result.exit_code == 0, f"CLI Error: {result.stderr}"
    output_file = input_dir / "compressed_output.txt"
    assert output_file.exists(), "compressed_output.txt not found in input directory"

    actual_output_text = output_file.read_text()
    expected_output = combined_content[:char_budget]
    assert actual_output_text == expected_output

    # Sanity check for truncation
    if len(combined_content) > len(actual_output_text) + 5:
         assert actual_output_text != combined_content


def test_compress_directory_with_output_dir_new(tmp_path):
    input_dir = tmp_path / "test_input_dir"
    input_dir.mkdir()
    output_dir = tmp_path / "test_output_dir"
    # output_dir.mkdir() # Command should create it

    (input_dir / "fileA.txt").write_text("Alpha content here.")
    (input_dir / "fileB.txt").write_text("Bravo content follows, adding more text.")

    combined_content = "Alpha content here.\n\nBravo content follows, adding more text."
    char_budget = 7

    result = runner.invoke(
        app,
        [
            "compress",
            "--dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--engine",
            "dummy_trunc", # Use "dummy_trunc" (engine ID directly)
            "--budget",
            str(char_budget),
        ],
    )

    assert result.exit_code == 0, f"CLI Error: {result.stderr}"
    assert output_dir.exists(), "Specified output directory was not created"
    output_file = output_dir / "compressed_output.txt"
    assert output_file.exists(), "compressed_output.txt not found in specified output directory"

    actual_output_text = output_file.read_text()
    expected_output = combined_content[:char_budget]
    assert actual_output_text == expected_output

    if len(combined_content) > len(actual_output_text) + 5:
         assert actual_output_text != combined_content


def test_compress_directory_recursive_pattern_new(tmp_path):
    input_dir = tmp_path / "test_input_dir"
    input_dir.mkdir()

    subdir = input_dir / "subdir"
    subdir.mkdir()

    (input_dir / "file1.txt").write_text("Text from file1 in root.")
    (input_dir / "another.md").write_text("Markdown file, should be ignored by *.txt pattern.")
    (input_dir / "file2.txt").write_text("Text from file2 in root.")
    (subdir / "file3.txt").write_text("Subdirectory text from file3.")
    (subdir / "notes.md").write_text("Another markdown in subdir.")

    # Glob order can be tricky. For predictability, let's assume Path.glob gives:
    # file1.txt, file2.txt, then subdir/file3.txt (or some consistent order)
    # To ensure order for the test, we can read them and sort them by name for combined_content

    paths = sorted(list(input_dir.glob("*.txt")) + list(subdir.glob("*.txt")))
    content_parts = [p.read_text() for p in paths]
    combined_content = "\n\n".join(content_parts)
    # This should result in:
    # "Text from file1 in root.\n\nText from file2 in root.\n\nSubdirectory text from file3."
    # (Order of file1/file2 might swap depending on glob, but file3 is last due to recursive glob)
    # For more stable test:
    # combined_content = "Text from file1 in root.\n\nText from file2 in root.\n\nSubdirectory text from file3."
    # Let's rely on the glob order for now but be aware.
    # The paths are: input_dir/file1.txt, input_dir/file2.txt, input_dir/subdir/file3.txt
    # A robust way:
    # combined_content = (input_dir / "file1.txt").read_text() + "\n\n" + \
    #                    (input_dir / "file2.txt").read_text() + "\n\n" + \
    #                    (subdir / "file3.txt").read_text()
    # This does not reflect the actual globbing and sorting done by the application.
    # The application uses: list(dir_path_obj.rglob(pattern))
    # So, the order depends on rglob. Let's simulate that order for `combined_content`.

    # Simulate rglob order for *.txt
    # Order of rglob is not guaranteed to be sorted alphabetically across directories.
    # However, within a directory, it's often alphabetical.
    # Let's assume file1.txt, file2.txt from root, then subdir/file3.txt
    # For the purpose of this test, let's construct combined_content assuming this order:
    # Root files first (sorted alphabetically), then subdirectory files (sorted alphabetically).
    file1_content = (input_dir / "file1.txt").read_text() # Text from file1 in root.
    file2_content = (input_dir / "file2.txt").read_text() # Text from file2 in root.
    file3_content = (subdir / "file3.txt").read_text()   # Subdirectory text from file3.
    # Based on rglob behavior, files in root are usually processed, then subdirectories.
    # Within each directory, it's often alphabetical.
    combined_content = f"{file1_content}\n\n{file2_content}\n\n{file3_content}"
    # combined_content should be:
    # "Text from file1 in root.\n\nText from file2 in root.\n\nSubdirectory text from file3."

    char_budget = 12

    result = runner.invoke(
        app,
        [
            "compress",
            "--dir",
            str(input_dir),
            "--recursive",
            "--pattern",
            "*.txt", # Explicitly test this
            "--engine",
            "dummy_trunc", # Use "dummy_trunc" (engine ID directly)
            "--budget",
            str(char_budget),
        ],
    )

    assert result.exit_code == 0, f"CLI Error: {result.stderr}"
    output_file = input_dir / "compressed_output.txt"
    assert output_file.exists(), "compressed_output.txt not found for recursive test"

    actual_output_text = output_file.read_text()
    expected_output = combined_content[:char_budget]
    assert actual_output_text == expected_output

    # This specific check for "markdown file" is covered by the exact match if expected_output is correct.
    # assert "markdown file" not in actual_output_text

    assert len(actual_output_text) > 0 # Should be true if char_budget > 0
    if len(combined_content) > len(actual_output_text) + 5: # Check if truncation happened
         assert actual_output_text != combined_content


# It's important to comment out or remove the old tests for directory compression
# as they test a different behavior (per-file output).

# def test_compress_directory_input_to_output_dir(tmp_path):
# ... (old test) ...
# def test_compress_directory_input_default_output(tmp_path):
# ... (old test) ...


@pytest.mark.parametrize(
    "input_args",
    [
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
        ["--text", "hi", "--file", "dummy.txt", "--dir", "dummy_dir"],  # All three
    ],
)
def test_compress_multiple_inputs_error(tmp_path, input_args):
    # Create dummy files/dirs if Typer's validation requires them to exist,
    # even if our target validation (mutual exclusivity) happens earlier.
    # This makes the test more robust against Typer's path validation order.
    if "--file" in input_args:
        idx = input_args.index("--file")
        f_path = tmp_path / input_args[idx + 1]
        f_path.touch()  # Create the dummy file
        input_args[idx + 1] = str(f_path)  # Update arg to actual path

    if "--dir" in input_args:
        idx = input_args.index("--dir")
        d_path = tmp_path / input_args[idx + 1]
        d_path.mkdir(exist_ok=True)  # Create the dummy dir
        input_args[idx + 1] = str(d_path)  # Update arg to actual path

    command = ["compress", "--engine", "none", "--budget", "50"] + input_args

    result = runner.invoke(app, command)

    assert (
        result.exit_code != 0
    ), f"CLI call with {input_args} did not fail as expected. Stderr: {result.stderr}"
    assert (
        "Specify exactly ONE of --text, --file, or --dir" in result.stderr.strip()
    )  # Adjusted to match actual
