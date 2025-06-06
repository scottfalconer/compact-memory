from typing import Union, List, Tuple, Any, Dict, Callable, Optional, Sequence # Added Sequence
import time # For CompressionTrace
import numpy as np

from compact_memory.engines.base import BaseCompressionEngine, CompressedMemory, CompressionTrace
from compact_memory.chunker import Chunker, SentenceWindowChunker # Default chunker

class ReadAgentGistEngine(BaseCompressionEngine):
    id: str = "readagent_gist"

    def __init__(
        self,
        chunker: Chunker | None = None,
        embedding_fn: Callable[[str | Sequence[str]], np.ndarray] | None = None,
        preprocess_fn: Callable[[str], str] | None = None,
        config: Optional[Dict[str, Any]] = None,
        local_llm_pipeline: Any = None,
        episode_token_limit: int = 500,
        gist_length: int = 100,
        gist_prompt_template: str = "Summarize the following text in about {gist_length} tokens: {text}",
        qa_prompt_template: str = "Based on the following context, answer the question. Context: {context} Question: {question}",
        lookup_prompt_template: str = "Based on the following summaries of pages and the question, which page(s) likely contain the details needed? Question: {question} Summaries: {summaries}",
        **kwargs: Any,
    ):
        # Ensure config is initialized before calling super if it's None and kwargs are provided
        if config is None:
            config = {} # Initialize config as an empty dict if None
        # Merge kwargs into config, giving precedence to kwargs if keys overlap.
        # This ensures that any specific engine parameters passed via kwargs are respected.
        config = {**config, **kwargs}


        super().__init__(chunker=chunker, embedding_fn=embedding_fn, preprocess_fn=preprocess_fn, config=config) # Pass merged config

        self.local_llm_pipeline = local_llm_pipeline
        # Retrieve parameters from self.config, allowing them to be overridden by what was passed in 'config' or 'kwargs'
        self.episode_token_limit = self.config.get('episode_token_limit', episode_token_limit)
        self.gist_length = self.config.get('gist_length', gist_length)
        self.gist_prompt_template = self.config.get('gist_prompt_template', gist_prompt_template)
        self.qa_prompt_template = self.config.get('qa_prompt_template', qa_prompt_template)
        self.lookup_prompt_template = self.config.get('lookup_prompt_template', lookup_prompt_template)

        if not callable(self.local_llm_pipeline):
            print(f"Warning: local_llm_pipeline is not callable. Gisting will be simulated.")

    def _paginate_episodes(self, text: str) -> List[str]:
        if not text: return []
        episodes = text.split('\n\n')
        episodes = [ep.strip() for ep in episodes if ep.strip()]
        return episodes if episodes else ([text.strip()] if text.strip() else [])

    def _generate_gist(self, episode_text: str) -> str:
        if not episode_text: return ""
        prompt = self.gist_prompt_template.format(gist_length=self.gist_length, text=episode_text)
        if callable(self.local_llm_pipeline):
            try:
                return self.local_llm_pipeline(prompt)
            except Exception as e:
                print(f"Error calling local_llm_pipeline: {e}")
                return f"Error generating gist. Input: {episode_text[:50]}..."
        return f"Simulated gist for episode: {episode_text[:50]}..."

    def _select_relevant_episodes(self, question: str, episode_gists: List[Tuple[int, str]]) -> List[int]:
        '''
        Selects relevant episode indices based on the question and gists using the LLM.
        episode_gists is a list of (episode_index, gist_text).
        Returns a list of selected episode indices.
        '''
        if not question or not episode_gists:
            return []

        formatted_summaries = []
        for index, gist_text in episode_gists:
            formatted_summaries.append(f"Page {index + 1}: {gist_text}") # 1-based for prompt

        summaries_str = "\n".join(formatted_summaries)

        prompt = self.lookup_prompt_template.format(question=question, summaries=summaries_str)

        selected_indices: List[int] = []
        if callable(self.local_llm_pipeline):
            try:
                response = self.local_llm_pipeline(prompt)
                # Attempt to parse response, e.g., "1, 3, 4" or "Page 1, Page 3"
                # This is a simplified parsing logic. Robust parsing is complex.
                raw_indices = response.replace("Page", "").split(',')
                for idx_str in raw_indices:
                    idx_str = idx_str.strip()
                    if idx_str.isdigit():
                        selected_indices.append(int(idx_str) - 1) # Convert back to 0-based
            except Exception as e:
                print(f"Error during LLM call or parsing in _select_relevant_episodes: {e}")
                # Fallback: if error, select no pages or first page as a guess.
                # For now, returning empty on error.
                # if episode_gists: selected_indices.append(0)
        else:
            # Simulation for subtask if LLM not callable
            if episode_gists: # Select first page if available during simulation
                 print("Simulating LLM call in _select_relevant_episodes, selecting first episode if available.")
                 selected_indices.append(0) # Return 0-based index directly for simulation consistency

        # Deduplicate, ensure indices are valid, and sort
        valid_indices = {idx for idx in selected_indices if 0 <= idx < len(episode_gists)}
        return sorted(list(valid_indices))


    def compress(
        self,
        text_or_chunks: Union[str, List[str]],
        llm_token_budget: int, # For now, treat as character budget
        **kwargs: Any,
    ) -> Tuple[CompressedMemory, CompressionTrace]:
        start_time = time.monotonic()
        trace_steps = []

        query: Optional[str] = kwargs.get("query", None) # Check for a query

        input_text: str
        if isinstance(text_or_chunks, list):
            input_text = " ".join(text_or_chunks)
            trace_steps.append({"type": "input_processing", "details": "Input converted from list of chunks to single string."})
        else:
            input_text = str(text_or_chunks)
            trace_steps.append({"type": "input_processing", "details": "Input is a single string."})

        episodes = self._paginate_episodes(input_text)
        trace_steps.append({"type": "pagination", "details": f"Text divided into {len(episodes)} episodes."})

        if not episodes:
            processing_time_ms = (time.monotonic() - start_time) * 1000
            trace = CompressionTrace(
                engine_name=self.id, strategy_params=self.config,
                input_summary={"original_length": len(input_text), "budget": llm_token_budget, "num_episodes": 0, "query": query},
                steps=trace_steps, output_summary={"compressed_length": 0, "text": ""}, processing_ms=processing_time_ms
            )
            return CompressedMemory(text=""), trace

        episode_gists_texts: List[str] = []
        for i, episode_text in enumerate(episodes):
            gist_start_time = time.monotonic() # Renamed for clarity from previous gist_processing_time_ms
            gist = self._generate_gist(episode_text)
            gist_processing_time_ms = (time.monotonic() - gist_start_time) * 1000
            episode_gists_texts.append(gist)
            trace_steps.append({
                "type": "gisting_episode",
                "details": {
                    "episode_index": i, "episode_preview": episode_text[:100] + "..." if len(episode_text) > 100 else episode_text,
                    "gist_preview": gist[:100] + "..." if len(gist) > 100 else gist,
                    "original_episode_length": len(episode_text), "gist_length": len(gist),
                    "processing_ms": gist_processing_time_ms
                }
            })

        final_text = ""

        if query:
            # QA Path
            trace_steps.append({"type": "task_detection", "details": f"Query detected: '{query[:100]}...'. Processing as QA."})

            # Prepare gists with their original indices for selection
            indexed_gists_for_selection: List[Tuple[int, str]] = list(enumerate(episode_gists_texts))

            selection_start_time = time.monotonic()
            selected_episode_indices = self._select_relevant_episodes(query, indexed_gists_for_selection)
            selection_processing_ms = (time.monotonic() - selection_start_time) * 1000
            trace_steps.append({
                "type": "episode_selection_for_qa",
                "details": {
                    "question": query[:100]+"...",
                    "num_gists_considered": len(indexed_gists_for_selection),
                    "selected_indices": selected_episode_indices,
                    "gists_preview_of_selected": {idx: episode_gists_texts[idx][:50]+"..." for idx in selected_episode_indices},
                    "processing_ms": selection_processing_ms
                }
            })

            context_parts = []
            for i, original_episode_text in enumerate(episodes): # Iterate through original episodes
                if i in selected_episode_indices:
                    context_parts.append(original_episode_text) # Use full original text
                else:
                    # Only include gist if it exists (i.e., i < len(episode_gists_texts))
                    if i < len(episode_gists_texts):
                        context_parts.append(episode_gists_texts[i]) # Use gist

            final_context = "\n\n---\n\n".join(context_parts)
            trace_steps.append({
                "type": "context_construction_for_qa",
                "details": {
                    "num_parts": len(context_parts),
                    "context_preview": final_context[:200]+"...",
                    "chars_full_text_in_context": sum(len(episodes[i]) for i in selected_episode_indices if i < len(episodes)),
                    "chars_gists_in_context": sum(len(episode_gists_texts[i]) for i in range(len(episodes)) if i not in selected_episode_indices and i < len(episode_gists_texts))
                }
            })

            qa_prompt = self.qa_prompt_template.format(context=final_context, question=query)

            qa_llm_start_time = time.monotonic()
            if callable(self.local_llm_pipeline):
                try:
                    answer = self.local_llm_pipeline(qa_prompt)
                    final_text = answer
                    qa_llm_processing_ms = (time.monotonic() - qa_llm_start_time) * 1000
                    trace_steps.append({"type": "qa_llm_call", "details": {"prompt_preview": qa_prompt[:200]+"...", "answer_preview": answer[:100]+"...", "processing_ms": qa_llm_processing_ms}})
                except Exception as e:
                    print(f"Error during QA LLM call: {e}")
                    final_text = "Error generating answer."
                    qa_llm_processing_ms = (time.monotonic() - qa_llm_start_time) * 1000
                    trace_steps.append({"type": "qa_llm_call_error", "details": {"error": str(e), "processing_ms": qa_llm_processing_ms}})
            else:
                final_text = f"Simulated answer for query: {query[:50]}..."
                qa_llm_processing_ms = (time.monotonic() - qa_llm_start_time) * 1000
                trace_steps.append({"type": "qa_llm_call", "details": "LLM pipeline not callable, using simulated answer.", "processing_ms": qa_llm_processing_ms})

            # Apply budget to the answer (simple truncation for now)
            if len(final_text) > llm_token_budget:
                original_answer_len = len(final_text)
                final_text = final_text[:llm_token_budget]
                trace_steps.append({"type": "answer_truncation", "details": f"Answer (length {original_answer_len}) exceeded budget ({llm_token_budget}). Truncated to {len(final_text)}."})

        else:
            # Summarization Path (as before)
            trace_steps.append({"type": "task_detection", "details": "No query detected. Processing as Summarization."})
            concatenated_gists = "\n\n---\n\n".join(episode_gists_texts)
            trace_steps.append({"type": "concatenation_for_summary", "details": f"Generated {len(episode_gists_texts)} gists. Total length: {len(concatenated_gists)}."}) # Corrected field name
            final_text = concatenated_gists
            if len(concatenated_gists) > llm_token_budget:
                original_summary_len = len(concatenated_gists)
                final_text = concatenated_gists[:llm_token_budget]
                trace_steps.append({"type": "summary_truncation", "details": f"Concatenated gists (length {original_summary_len}) exceeded budget ({llm_token_budget}). Truncated to {len(final_text)}."})
            else:
                trace_steps.append({"type": "budget_check", "details": f"Concatenated gists (length {len(concatenated_gists)}) within budget ({llm_token_budget}). No truncation needed."})

        processing_time_ms = (time.monotonic() - start_time) * 1000
        output_summary = {
            "compressed_length": len(final_text), "num_episodes": len(episodes),
            "num_gists": len(episode_gists_texts), "query_processed": bool(query),
            "final_text_preview": final_text[:100] + "..." if len(final_text) > 100 else final_text
        }

        trace = CompressionTrace(
            engine_name=self.id, strategy_params=self.config,
            input_summary={"original_length": len(input_text), "budget": llm_token_budget, "query": query},
            steps=trace_steps, output_summary=output_summary, processing_ms=processing_time_ms,
            final_compressed_object_preview=final_text[:100] # From base class
        )
        return CompressedMemory(text=final_text, metadata={"num_episodes": len(episodes), "num_gists": len(episode_gists_texts), "query_processed": bool(query), "final_answer_preview_if_qa": final_text[:100] if query else None}), trace

    def _compress_chunk(self, chunk_text: str) -> str:
        '''
        Overrides BaseCompressionEngine's method.
        For ReadAgent, a "chunk" from the chunker is treated as an "episode".
        This method generates and returns the gist for that episode.
        The base ingest method will then store this gist and its embedding.
        '''
        if not chunk_text.strip():
            # If the chunk is empty or just whitespace, return empty string.
            # _generate_gist might also handle this, but good to be defensive.
            return ""

        # Treat the incoming chunk_text as an episode and generate its gist
        gist = self._generate_gist(chunk_text)
        # print(f"Debug ReadAgentGistEngine: _compress_chunk received chunk (len {len(chunk_text)}), generated gist (len {len(gist)})")
        return gist

    # No need to override ingest if self.chunker produces episodes
    # and _compress_chunk produces gists. The base ingest will then:
    # 1. Use self.chunker to get "episodes" (as raw_chunks)
    # 2. Call our _compress_chunk (which is _generate_gist) for each -> processed_chunks are gists
    # 3. Embed and store these gists.

    # Regarding the `recall` method:
    # The `ReadAgentGistEngine` relies on the `recall` method inherited from
    # `BaseCompressionEngine` for retrieving stored gists.
    #
    # How it works:
    # 1. During `ingest(text)`, the input `text` is chunked by `self.chunker`.
    #    Each chunk is treated as an "episode" by this engine.
    # 2. Our overridden `_compress_chunk(episode_text)` is called for each episode.
    #    This method generates a `gist` for the `episode_text`.
    # 3. The `BaseCompressionEngine.ingest` method then takes these `gists` (as they are
    #    the output of `_compress_chunk`), computes their embeddings using `self.embedding_fn`,
    #    and stores both the gist text and its embedding.
    # 4. Consequently, when `BaseCompressionEngine.recall(query, top_k=...)` is called:
    #    - It embeds the `query`.
    #    - It performs a similarity search against the stored embeddings (which are embeddings of gists).
    #    - It returns a list of dictionaries, where each `{"text": ...}` field contains
    #      the text of a retrieved gist.
    #
    # This fulfills the requirement of retrieving relevant gists based on vector similarity
    # to the query. No specific override of `recall` is needed for this core functionality.
    # Advanced recall logic (e.g., LLM-based re-ranking of gists) could be implemented
    # here if required in the future.

    # Other methods like recall
