# Tutorial 2: Step-by-Step - Building a Simple compression engine
This tutorial walks you through creating a basic compression engine for Compact Memory. It complements the main \`ENGINE_DEVELOPMENT.md\` guide by providing a hands-on example. We'll build a engine that keeps only the first N sentences of a text.
## Prerequisites
*   `Compact Memory installed (\`pip install compact-memory\`).`
*   `Familiarity with Python.`
*   `(Optional but recommended)` install the spaCy extra for robust sentence tokenization:
    `pip install compact-memory[spacy] && python -m spacy download en_core_web_sm`
## Goal
Our engine, \`FirstNSentencesEngine\`, will:
1.  `Take a piece of text as input.`
2.  `Identify sentences in the text.`
3.  `Keep the first N sentences, where N is a configurable parameter.`
4.  `Try to adhere to a given token budget, potentially returning fewer than N sentences if the budget is too small.`
## Steps
### 1. Define the Engine Class
Create a Python file for your engine, for example, \`my_engines.py\`.
```python
from typing import Union, List, Tuple, Any
from compact_memory.engines import BaseCompressionEngine, CompressedMemory, CompressionTrace
from compact_memory.token_utils import get_tokenizer, token_count

try:
    import spacy
    nlp_spacy = spacy.load("en_core_web_sm")
    print("spaCy loaded for sentence tokenization.")
except ImportError:
    nlp_spacy = None
    print("spaCy not found. Sentence splitting will be very basic (newline-based).")

class FirstNSentencesEngine(BaseCompressionEngine):
    id: str = "first_n_sentences"

    def __init__(self, num_sentences: int = 3):
        super().__init__()
        self.num_sentences_to_keep = num_sentences
        print(f"FirstNSentencesEngine initialized to keep {self.num_sentences_to_keep} sentences.")

    def _split_into_sentences(self, text: str) -> List[str]:
        if nlp_spacy:
            doc = nlp_spacy(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            return [s.strip() for s in text.splitlines() if s.strip()]

    def compress(
        self,
        text_or_chunks: Union[str, List[str]],
        llm_token_budget: int,
        **kwargs: Any,
    ) -> Tuple[CompressedMemory, CompressionTrace]:
        input_text: str
        if isinstance(text_or_chunks, list):
            input_text = " ".join(text_or_chunks)
        else:
            input_text = str(text_or_chunks)

        tokenizer = kwargs.get("tokenizer")
        tokenizer_for_budgeting = tokenizer if tokenizer else lambda t: t.split()
        if not tokenizer:
            print("Warning: No tokenizer provided. Using word count for budgeting (less accurate).")

        original_token_count_val = token_count(tokenizer_for_budgeting, input_text)

        trace_steps = [{'type': 'input_processing', 'details': {'input_type': type(text_or_chunks).__name__, 'original_length': len(input_text), 'original_tokens': original_token_count_val, 'budget_type': 'tokens' if tokenizer else 'words (fallback)', 'requested_budget': llm_token_budget}}]
        sentences = self._split_into_sentences(input_text)
        trace_steps.append({'type': 'sentence_splitting', 'details': {'num_sentences_found': len(sentences), 'method': 'spaCy' if nlp_spacy else 'newline split'}})

        selected_sentences = []
        current_tokens = 0
        sentences_taken = 0
        for i, sentence in enumerate(sentences):
            if i >= self.num_sentences_to_keep:
                trace_steps.append({'type': 'selection_stopped', 'details': {'reason': f'Reached configured limit of {self.num_sentences_to_keep} sentences.', 'sentences_selected': sentences_taken}}); break
            sentence_token_count = token_count(tokenizer_for_budgeting, sentence)
            if (current_tokens + sentence_token_count) <= llm_token_budget:
                selected_sentences.append(sentence); current_tokens += sentence_token_count; sentences_taken += 1
                trace_steps.append({'type': 'sentence_selected', 'details': {'sentence_index': i, 'sentence_preview': sentence[:50] + '...', 'sentence_tokens': sentence_token_count, 'cumulative_tokens': current_tokens}})
            else:
                trace_steps.append({'type': 'sentence_skipped_budget', 'details': {'sentence_index': i, 'sentence_preview': sentence[:50] + '...', 'sentence_tokens': sentence_token_count, 'reason': f'Adding sentence would exceed token budget ({current_tokens + sentence_token_count} > {llm_token_budget}).'}}); break

        compressed_text = " ".join(selected_sentences)
        final_token_count = token_count(tokenizer_for_budgeting, compressed_text)
        trace = CompressionTrace(engine_name=self.id, engine_params={'num_sentences': self.num_sentences_to_keep}, input_summary={'original_tokens': original_token_count_val, 'num_input_sentences': len(sentences)}, steps=trace_steps, output_summary={'compressed_tokens': final_token_count, 'num_output_sentences': len(selected_sentences)}, processing_ms=0.0, final_compressed_object_preview=compressed_text[:70])
        return CompressedMemory(text=compressed_text, metadata={'sentences_kept': len(selected_sentences)}), trace
```
### 2. Register and Test the Engine
Now, let's test our new engine. You can do this in the same \`my_engines.py\` file or a separate test script.
```python
from compact_memory.registry import register_compression_engine
from compact_memory.token_utils import get_tokenizer

if __name__ == "__main__":
    register_compression_engine(FirstNSentencesEngine.id, FirstNSentencesEngine)

    print("\n--- Test Case 1: Basic ---")
    engine_test1 = FirstNSentencesEngine(num_sentences=2)
    sample_text1 = "This is the first sentence. It provides an introduction.\nThis is the second sentence, offering more details.\nThis is a third sentence, which should be excluded by this configuration.\nAnd a fourth one, also to be excluded."
    try: test_tokenizer = get_tokenizer("gpt2")
    except ImportError: test_tokenizer = str.split
    compressed_mem1, trace1 = engine_test1.compress(sample_text1, llm_token_budget=50, tokenizer=test_tokenizer)
    print(f"Original Text:\n{sample_text1}")
    print(f"Compressed Text (Target 2 sentences, Budget 50):\n{compressed_mem1.text}")
    print(f"Tokens: {token_count(test_tokenizer, compressed_mem1.text)}")
    print(f"Metadata: {compressed_mem1.metadata}")

    print("\n--- Test Case 2: Budget Constraint ---")
    engine_test2 = FirstNSentencesEngine(num_sentences=5)
    sample_text2 = "Sentence one is short. Sentence two is also quite brief.\nSentence three is a bit longer and might push the budget.\nSentence four is definitely very long and elaborate, probably too much for a small budget.\nSentence five would be next if budget allows."
    compressed_mem2, trace2 = engine_test2.compress(sample_text2, llm_token_budget=20, tokenizer=test_tokenizer)
    print(f"Original Text:\n{sample_text2}")
    print(f"Compressed Text (Target 5 sentences, Budget 20):\n{compressed_mem2.text}")
    print(f"Tokens: {token_count(test_tokenizer, compressed_mem2.text)}")
    print(f"Metadata: {compressed_mem2.metadata}")

    if nlp_spacy:
        print("\n--- Test Case 3: No spaCy (Fallback Sentence Splitting) ---")
        original_nlp_spacy = nlp_spacy; nlp_spacy = None # type: ignore
        engine_test3 = FirstNSentencesEngine(num_sentences=2)
        sample_text3 = "First line is a sentence.\nSecond line is another."
        compressed_mem3, trace3 = engine_test3.compress(sample_text3, llm_token_budget=30, tokenizer=test_tokenizer)
        print(f"Original Text:\n{sample_text3}")
        print(f"Compressed Text (Fallback, Target 2 sentences, Budget 30):\n{compressed_mem3.text}")
        print(f"Tokens: {token_count(test_tokenizer, compressed_mem3.text)}")
        print(f"Metadata: {compressed_mem3.metadata}")
        nlp_spacy = original_nlp_spacy
```
Explanation of the Code:
*   `FirstNSentencesEngine(BaseCompressionEngine)`: Inherits from the base `BaseCompressionEngine`.
*   `id`: A unique string identifier for the engine.
*   `__init__`: Takes `num_sentences` to configure how many sentences to keep.
*   `_split_into_sentences`: Uses `spaCy` if available for robust sentence splitting, otherwise falls back to splitting by newlines.
*   `compress`:
    *   Handles string or list-of-strings input.
    *   Uses a provided `tokenizer` or a simple fallback for budgeting.
    *   Iterates through sentences, adding them to `selected_sentences` if they fit within `num_sentences_to_keep` and `llm_token_budget`.
    *   Builds a `CompressionTrace` to log its actions.
    *   Returns a `CompressedMemory` object.
*   Test cases demonstrate basic usage, budget constraints, and the spaCy fallback.
### 3. Using the Engine via Compact Memory CLI or API
Once your engine is defined and you can import it (and optionally register it if not using a plugin mechanism for broader discovery), you can use it.
*   CLI:
To use via CLI, you'd typically package this engine (see \`SHARING_ENGINES.md\`). If \`my_engines.py\` was in a discoverable plugin directory, you could run:
```bash
compact-memory compress --text "Your long text here..." --engine first_n_sentences --budget 50 --engine-params '{"num_sentences": 2}'
```
*   Python API:
```python
from compact_memory import get_compression_engine
# Assuming FirstNSentencesEngine is registered or imported
# For example, if you ran the __main__ block from my_engines.py or imported it.

# Get the engine class (if registered)
try:
    EngineClass = get_compression_engine("first_n_sentences")
    # Instantiate with parameters
    my_engine = EngineClass(num_sentences=2)

    text_to_compress = "This is a full sentence. This is another sentence that provides context. This third sentence is important for details. A fourth sentence might be too much for the budget. The fifth and final sentence concludes the example."

    # Get a tokenizer
    try:
        tokenizer = get_tokenizer("gpt2")
    except ImportError:
        tokenizer = str.split # Fallback

    compressed_output, trace_output = my_engine.compress(
        text_to_compress,
        llm_token_budget=20, # Example budget
        tokenizer=tokenizer
    )
    print("\n--- API Usage Test ---")
    print(f"Compressed Text: {compressed_output.text}")
    print(f"Tokens: {token_count(tokenizer, compressed_output.text)}")
    print(f"Trace: {trace_output.engine_name}, Steps: {len(trace_output.steps)}")
except KeyError:
    print("Engine 'first_n_sentences' not found. Ensure it's registered for API usage outside its defining script.")

```
## Further Development
This basic engine can be extended:
*   `More Sophisticated Fallback: If spaCy isn't available, use a regex-based sentence splitter.`
*   `Smarter Budgeting: Instead of just stopping, it could try to include a partial last sentence if it fits most of the budget, or use more precise token-aware truncation for the last included sentence.`
*   `Configurable Sentence Delimiter: Allow users to specify custom sentence delimiters.`
*   `Min/Max Sentences: Add parameters for minimum sentences to keep, regardless of budget (if feasible), or a hard maximum even if budget allows more.`
## Conclusion
This tutorial demonstrated building a simple \`FirstNSentencesEngine\`, from defining the class to testing it. It illustrates the core components of a \`BaseCompressionEngine\` and how to integrate it with Compact Memory's utilities. For more advanced features like learnable components, detailed tracing, and packaging, refer to the main \`ENGINE_DEVELOPMENT.md\` guide and the \`SHARING_ENGINES.md\` guide.
