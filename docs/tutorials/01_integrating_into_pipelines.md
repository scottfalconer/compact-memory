# Tutorial 1: Integrating Compact Memory into Your LLM Pipeline

This tutorial demonstrates how to programmatically use an existing compression engine from Compact Memory to compress text and then integrate that compressed text into a pipeline with a Large Language Model (LLM). This is ideal for users who want to apply pre-built engines to optimize context in their applications.

**Scenario:** You have a long piece of text (e.g., a document, an article, or extensive user history) that you need to feed to an LLM, but it exceeds the LLM\'s context window or your desired token budget. You want to use Compact Memory to compress this text before sending it to the LLM.

## Prerequisites

1.  **Compact Memory Installed:** Ensure you have Compact Memory installed.
    ```bash
    pip install compact-memory
    ```
2.  **LLM Access:** You should have access to an LLM, either via an API (like OpenAI, Gemini) or a locally running model. This tutorial will use a placeholder for the LLM call.
3.  **(Optional) Tokenizer:** For accurate token counting and budget management, it\'s good to have a tokenizer like `tiktoken`.
    ```bash
    pip install tiktoken
    ```

## Steps

### 1. Import Necessary Modules

First, let\'s import the required components from Compact Memory and any other libraries.

```python
from compact_memory import get_compression_engine
from compact_memory.engines import CompressedMemory, BaseCompressionEngine
from compact_memory.token_utils import get_tokenizer, token_count # For token counting
import os # For API keys, if needed

# Placeholder for your actual LLM calling function
def my_llm_call(prompt: str, model_name: str = "gpt-4.1-nano") -> str:
    print(f"---- Sending to LLM ({model_name}) ----")
    print(prompt)
    print("---- End of LLM Prompt ----")
    # In a real scenario, you would make an API call here, e.g.:
    # from openai import OpenAI
    # client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    # response = client.chat.completions.create(
    #     model=model_name,
    #     messages=[{"role": "user", "content": prompt}]
    # )
    # return response.choices[0].message.content
    return f"LLM response based on the provided context about the text."
```

### 2. Select and Load a compression engine

Compact Memory comes with built-in engines, and you might have installed others as plugins. You can list available engines using the CLI (`compact-memory dev list-engines`). For this example, we\'ll use the `first_last` engine, which simply takes the beginning and end of the text.

```python
# Choose an engine ID
# For this example, \'first_last\' is simple. Other options could be \'truncate\',
# or more advanced ones if available (e.g., \'prototype\' if you have an agent store).
engine_id = "first_last"

try:
    # Get the engine class
    EngineClass = get_compression_engine(engine_id)
    # Instantiate the engine
    compression_engine: BaseCompressionEngine = EngineClass()
    print(f"Successfully loaded engine: {engine_id}")
except KeyError:
    print(f"Error: Engine \'{engine_id}\' not found. Ensure it\'s a valid built-in or plugin engine.")
    exit()
```

### 3. Prepare Your Text and Define Budget

Let\'s define the long text we want to compress and the token budget for the compressed output.

```python
long_text_to_compress = """\
The Industrial Revolution, which began in Great Britain in the late 18th century and later spread to other parts of the world,
was a period of major technological advancement. Key innovations included the steam engine, which revolutionized transportation
and manufacturing; the power loom, which mechanized textile production; and the cotton gin, which significantly increased
the efficiency of cotton processing. These changes led to unprecedented economic growth, urbanization, and societal shifts.
Factories became the new centers of work, attracting large populations to cities. This rapid urbanization, however,
also brought challenges such as overcrowding, poor sanitation, and harsh working conditions for many.
The era also saw the rise of new economic theories, such as capitalism, and social movements advocating for workers\' rights.
Transportation improved dramatically with the development of railways and steamships, facilitating trade and migration.
Communication was also enhanced by inventions like the telegraph. The impact of the Industrial Revolution was profound,
reshaping almost every aspect of human life and setting the stage for the modern world. Its legacy includes not only
technological progress but also ongoing debates about its social and environmental consequences. Further developments in
the Second Industrial Revolution, from the late 19th to early 20th centuries, brought about advancements like electricity,
the internal combustion engine, and mass production techniques.
"""

# Define your target token budget for the compressed text
# This depends on your LLM\'s context window and your specific needs
token_budget = 75 # Aim for approximately 75 tokens
```

### 4. Obtain a Tokenizer (Recommended)

Using a tokenizer helps the engine adhere to the `token_budget` more accurately and allows you to verify token counts.

```python
try:
    # Using "gpt2" tokenizer as an example, common for many models
    tokenizer = get_tokenizer("gpt2")
    print("Using tiktoken (gpt2) tokenizer.")
except ImportError:
    # Fallback if tiktoken is not installed or fails
    tokenizer = str.split # Simple word-based tokenizer (less accurate)
    print("tiktoken not found. Using simple word split as a fallback tokenizer (less accurate for token budgeting).")

# Optional: Check original token count
original_token_count = token_count(tokenizer, long_text_to_compress)
print(f"Original text approximate token count: {original_token_count}")
```

### 5. Compress the Text

Now, use the loaded engine to compress your text.

```python
# The compress method returns a CompressedMemory object and a CompressionTrace object
compressed_memory, trace = compression_engine.compress(
    text_or_chunks=long_text_to_compress,
    llm_token_budget=token_budget,
    tokenizer=tokenizer # Pass the tokenizer to the engine
)

# Access the compressed text
final_compressed_text = compressed_memory.text

# Optional: Review trace information and token counts
print(f"--- Compression Results ---")
print(f"Engine Used: {trace.engine_name}")
if trace.engine_params:
    print(f"Engine Parameters: {trace.engine_params}")
print(f"Compressed Text: \'{final_compressed_text}\'")
compressed_token_count = token_count(tokenizer, final_compressed_text)
print(f"Compressed text approximate token count: {compressed_token_count}")
if trace.processing_ms is not None:
    print(f"Compression Time: {trace.processing_ms:.2f} ms")
```

### 6. Use the Compressed Text in Your LLM Pipeline

With the text compressed, you can now use it in your prompt to the LLM.

```python
# Define your question or task for the LLM
user_question = "Based on the provided text, what were the key impacts of the Industrial Revolution?"

# Construct the prompt using the compressed text as context
prompt_for_llm = f"""\
Context:
{final_compressed_text}

Question:
{user_question}

Answer:
"""

# Call your LLM
llm_response = my_llm_call(prompt_for_llm)

print(f"--- LLM Response ---")
print(llm_response)
```

## Complete Example Script

```python
from compact_memory import get_compression_engine
from compact_memory.engines import CompressedMemory, BaseCompressionEngine
from compact_memory.token_utils import get_tokenizer, token_count
import os

# Placeholder for your actual LLM calling function
def my_llm_call(prompt: str, model_name: str = "gpt-4.1-nano") -> str:
    print(f"---- Sending to LLM ({model_name}) ----")
    print(prompt)
    print("---- End of LLM Prompt ----")
    # In a real scenario, you would make an API call here
    return f"LLM response based on the provided context about the text."

def run_pipeline():
    # 1. Select and Load a compression engine
    engine_id = "first_last"
    try:
        EngineClass = get_compression_engine(engine_id)
        compression_engine: BaseCompressionEngine = EngineClass()
        print(f"Successfully loaded engine: {engine_id}")
    except KeyError:
        print(f"Error: Engine \'{engine_id}\' not found.")
        return

    # 2. Prepare Your Text and Define Budget
    long_text_to_compress = """\
The Industrial Revolution, which began in Great Britain in the late 18th century and later spread to other parts of the world,
was a period of major technological advancement. Key innovations included the steam engine, which revolutionized transportation
and manufacturing; the power loom, which mechanized textile production; and the cotton gin, which significantly increased
the efficiency of cotton processing. These changes led to unprecedented economic growth, urbanization, and societal shifts.
Factories became the new centers of work, attracting large populations to cities. This rapid urbanization, however,
also brought challenges such as overcrowding, poor sanitation, and harsh working conditions for many.
The era also saw the rise of new economic theories, such as capitalism, and social movements advocating for workers\' rights.
Transportation improved dramatically with the development of railways and steamships, facilitating trade and migration.
Communication was also enhanced by inventions like the telegraph. The impact of the Industrial Revolution was profound,
reshaping almost every aspect of human life and setting the stage for the modern world. Its legacy includes not only
technological progress but also ongoing debates about its social and environmental consequences. Further developments in
the Second Industrial Revolution, from the late 19th to early 20th centuries, brought about advancements like electricity,
the internal combustion engine, and mass production techniques.
"""
    token_budget = 75

    # 3. Obtain a Tokenizer
    try:
        tokenizer = get_tokenizer("gpt2")
        print("Using tiktoken (gpt2) tokenizer.")
    except ImportError:
        tokenizer = str.split
        print("tiktoken not found. Using simple word split as a fallback tokenizer.")

    original_token_count = token_count(tokenizer, long_text_to_compress)
    print(f"Original text approximate token count: {original_token_count}")

    # 4. Compress the Text
    compressed_memory, trace = compression_engine.compress(
        text_or_chunks=long_text_to_compress,
        llm_token_budget=token_budget,
        tokenizer=tokenizer
    )
    final_compressed_text = compressed_memory.text

    print(f"--- Compression Results ---")
    print(f"Engine Used: {trace.engine_name}")
    print(f"Compressed Text: \'{final_compressed_text}\'")
    compressed_token_count = token_count(tokenizer, final_compressed_text)
    print(f"Compressed text approximate token count: {compressed_token_count}")

    # 5. Use the Compressed Text in Your LLM Pipeline
    user_question = "Based on the provided text, what were the key impacts of the Industrial Revolution?"
    prompt_for_llm = f"Context:\n{final_compressed_text}\n\nQuestion:\n{user_question}\n\nAnswer:"

    llm_response = my_llm_call(prompt_for_llm)
    print(f"--- LLM Response ---")
    print(llm_response)

if __name__ == "__main__":
    run_pipeline()
```

## Conclusion

This tutorial showed how to integrate Compact Memory into a Python-based LLM pipeline. By loading an engine, compressing text to a specified budget, and then using that compressed text as context for an LLM, you can effectively manage longer inputs and optimize your LLM interactions. You can adapt this workflow with different engines and integrate it into more complex applications.
