from compact_memory import get_compression_engine
# from compact_memory.token_utils import get_tokenizer # This function does not exist in token_utils.py

# Ensure experimental engines are available if needed,
# though 'first_last' should be standard.
# from compact_memory.engines import enable_all_experimental_engines
# enable_all_experimental_engines()

try:
    engine_name = "first_last"
    engine = get_compression_engine(engine_name)()
    print(f"Successfully loaded engine: {engine_name}")

    text_to_compress = "This is a very long document that we want to summarize using the Python API. It has several sentences, and we expect the first_last engine to pick the beginning and the end of this text, according to the token budget."
    budget = 20 # tokens

    # Get a tokenizer to verify token count (optional, for deeper inspection)
    # tokenizer = get_tokenizer("gpt2") # or any other tokenizer
    # tokens_before = len(tokenizer.encode(text_to_compress))
    # print(f"Tokens before compression: {tokens_before}")

    compressed_memory, trace = engine.compress(text_to_compress, llm_token_budget=budget)

    print(f"Original text: '{text_to_compress}'")
    print(f"Compressed text: '{compressed_memory.text}'")
    # tokens_after = len(tokenizer.encode(compressed_memory.text))
    # print(f"Tokens after compression: {tokens_after}")
    # print(f"Trace: {trace}")

    expected_start = "This is a very long document that we want to summarize using the Python API."
    # Depending on the tokenizer and budget, the exact end part might vary.
    # We are mostly checking that it runs and produces shorter output.
    if compressed_memory.text.startswith(expected_start.split('.')[0]) and len(compressed_memory.text) < len(text_to_compress):
        print("Python API test successful: Compression occurred and output seems reasonable.")
    else:
        print("Python API test potentially failed: Output does not look as expected.")

except Exception as e:
    print(f"Error during Python API test: {e}")
    import traceback
    traceback.print_exc()
