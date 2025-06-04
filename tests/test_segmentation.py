from pathlib import Path
import pytest # Added
from unittest.mock import patch # Added
import sys # Added for sys.modules manipulation in langchain test

from examples.chunking import ( # Updated imports
    agentic_split,
    _sentences,
    newline_splitter,
    tiktoken_fixed_size_splitter,
    langchain_recursive_splitter
)


def test_agentic_split_long_text():
    text = Path(Path(__file__).parent / "data" / "constitution.txt").read_text()
    chunks = agentic_split(text, max_tokens=40)
    assert len(chunks) > 5
    assert all(len(c.split()) <= 80 for c in chunks)


def test_spacy_sentence_segmentation_abbreviation():
    text = "Dr. Smith arrived at 5 p.m. He said hello. Goodbye."
    sents = _sentences(text)
    assert len(sents) == 3

# --- New tests below ---

def test_newline_splitter():
    text = "Hello\nWorld\nThis is a test."
    chunks = newline_splitter(text)
    assert chunks == ["Hello", "World", "This is a test."]
    text_no_newline = "Hello World"
    assert newline_splitter(text_no_newline) == ["Hello World"]
    text_empty = ""
    # Per Python's str.split behavior, splitting an empty string results in a list containing one empty string.
    # If the desired behavior for an empty input was an empty list, the function would need adjustment.
    assert newline_splitter(text_empty) == [""]
    text_trailing_newline = "Hello\nWorld\n"
    assert newline_splitter(text_trailing_newline) == ["Hello", "World", ""]


def test_tiktoken_fixed_size_splitter_basic():
    # This test assumes tiktoken is installed in the test environment
    try:
        import tiktoken
        text = "This is a sample text for testing tiktoken splitter. It has several sentences and words."
        # Test with a small chunk size to ensure splitting
        chunks = tiktoken_fixed_size_splitter(text, chunk_size=10, model_name="cl100k_base")
        assert len(chunks) > 1
        # Ensure individual chunks (when re-encoded) are within or slightly over due to token boundaries.
        encoding = tiktoken.get_encoding("cl100k_base")
        for chunk in chunks:
            # Adding a small buffer because decode(encode(text)) might not be perfectly identity
            # and tokenization can lead to slightly more tokens after decode/re-encode of a segment.
            # The core check is that it attempts to stay near chunk_size.
            assert len(encoding.encode(chunk)) <= chunk_size + 5 # Allow some leeway
        assert "".join(chunks) == text # Check if text is preserved
    except ImportError:
        pytest.skip("tiktoken not installed, skipping tiktoken_fixed_size_splitter basic test")

def test_tiktoken_fixed_size_splitter_unknown_model():
    try:
        import tiktoken # Check if tiktoken is available
        text = "Another test sentence for the unknown model fallback."
        # Using a clearly non-existent model name to test fallback to cl100k_base
        chunks = tiktoken_fixed_size_splitter(text, chunk_size=5, model_name="non_existent_model_blah_blah_12345")
        assert len(chunks) >= 1
        assert "".join(chunks) == text
        # Verify it used a fallback encoding (e.g. cl100k_base) by checking tokenization properties if needed,
        # but for now, just ensuring it runs and preserves text is sufficient.
        encoding = tiktoken.get_encoding("cl100k_base") # assumes this is the fallback
        tokens = encoding.encode(text)
        expected_num_chunks = (len(tokens) + 5 - 1) // 5 # ceil(len(tokens)/chunk_size)
        assert len(chunks) == expected_num_chunks

    except ImportError:
        pytest.skip("tiktoken not installed, skipping tiktoken_fixed_size_splitter unknown model test")

@patch('importlib.import_module')
def test_tiktoken_splitter_import_error(mock_import_module):
    # Simulate tiktoken not being importable
    mock_import_module.side_effect = ImportError("No module named tiktoken")

    # Need to ensure that 'tiktoken' is not already in sys.modules from a previous successful import
    # in another test, or that the function re-tries the import.
    # The function `tiktoken_fixed_size_splitter` tries `import tiktoken` directly.
    # If `tiktoken` was successfully imported by a previous test in the same session,
    # this mock might not prevent its use unless the module itself is unloaded or the specific
    # lookup is patched.
    # A more robust way for functions that import *inside* them is to patch `sys.modules` temporarily
    # or ensure the import mechanism they use is what's patched.
    # Given the current implementation of tiktoken_fixed_size_splitter,
    # patching importlib.import_module for 'tiktoken' should work if it's the first time it's called
    # or if the module is reloaded. Let's refine this if issues arise.

    original_tiktoken_in_sys_modules = 'tiktoken' in sys.modules
    if original_tiktoken_in_sys_modules:
        # If tiktoken was already imported (e.g. by a previous test), this mock needs to be more targeted.
        # We can temporarily remove it from sys.modules for this test's scope.
        # However, the function itself does `import tiktoken`. `patch.dict` is better for this.
        pass # Will proceed with importlib.import_module patch, should be effective for direct `import tiktoken`

    with patch.dict('sys.modules', {'tiktoken': None}): # Ensure tiktoken is seen as not imported
        with pytest.raises(ImportError, match="tiktoken library not found"):
            tiktoken_fixed_size_splitter("test text for import error", chunk_size=10)

    # If we did more complex sys.modules manipulation, restore original_tiktoken_in_sys_modules here.


def test_langchain_recursive_splitter_basic():
    # This test assumes langchain is installed
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter # To check if actually available
        text = "This is a sample text for testing LangChain splitter. It has several sentences and words, and we expect it to be split."
        chunks = langchain_recursive_splitter(text, chunk_size=20, chunk_overlap=5)
        assert len(chunks) > 1
        # Basic check: all chunks should be strings
        assert all(isinstance(chunk, str) for chunk in chunks)
        # A more robust check would be to see if the text is preserved after joining,
        # but RecursiveCharacterTextSplitter can add separators or alter whitespace slightly
        # depending on its configuration, so direct join might not equal original.
        # For now, checking type and that splitting occurred is the main goal.
    except ImportError:
        pytest.skip("langchain not installed, skipping langchain_recursive_splitter basic test")

@patch('importlib.import_module')
def test_langchain_splitter_import_error(mock_import_module):
    # This mock targets the `from langchain.text_splitter import RecursiveCharacterTextSplitter`
    # within the `langchain_recursive_splitter` function.
    def import_side_effect(name, *args, **kwargs):
        if name == 'langchain.text_splitter':
            raise ImportError("No module named langchain.text_splitter")
        # Attempt to fall back to the original import mechanism for other modules
        # This might require careful handling if other modules are also imported by the function under test
        # For this specific function, only one import is made.
        try:
            return __builtins__['__import__'](name, *args, **kwargs)
        except ModuleNotFoundError: # pragma: no cover
             # This fallback might not always work depending on how importlib.import_module is used elsewhere.
             # For this specific test, if 'langchain.text_splitter' is requested, it raises.
             # Any other import would be unexpected for the function being tested.
            raise ModuleNotFoundError(f"No module named '{name}' found by mock fallback.")


    mock_import_module.side_effect = import_side_effect

    # To ensure the test is effective even if langchain was imported by another test,
    # we can use patch.dict to temporarily remove the specific module from sys.modules.
    with patch.dict('sys.modules', {'langchain.text_splitter': None, 'langchain': None}):
        with pytest.raises(ImportError, match="langchain library not found"):
            langchain_recursive_splitter("test text for import error")
