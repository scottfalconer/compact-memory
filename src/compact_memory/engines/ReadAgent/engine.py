from typing import (
    Union,
    List,
    Tuple,
    Any,
    Dict,
    Callable,
    Optional,
    Sequence,
)  # Added Sequence
import time  # For CompressionTrace
import numpy as np

from compact_memory.llm_providers_abc import LLMProvider  # Added
from compact_memory.engines.base import (
    BaseCompressionEngine,
    CompressedMemory,
    CompressionTrace,
)
from compact_memory.chunker import Chunker, SentenceWindowChunker  # Default chunker
from compact_memory.token_utils import token_count, truncate_text
from compact_memory.engines.registry import register_compression_engine

try:  # pragma: no cover - optional dependency
    import tiktoken

    _DEFAULT_TOKENIZER = tiktoken.get_encoding("gpt2")
except Exception:  # pragma: no cover - optional dependency
    _DEFAULT_TOKENIZER = None


class ReadAgentGistEngine(BaseCompressionEngine):
    id: str = "read_agent_gist"

    def __init__(
        self,
        chunker: Chunker | None = None,
        embedding_fn: Callable[[str | Sequence[str]], np.ndarray] | None = None,
        preprocess_fn: Callable[[str], str] | None = None,
        config: Optional[Dict[str, Any]] = None,
        llm_provider: Optional[
            LLMProvider
        ] = None,  # Changed local_llm_pipeline to llm_provider
        episode_token_limit: int = 500,
        gist_length: int = 100,  # This will be used as max_new_tokens for gisting
        gist_prompt_template: str = "Summarize the following text in about {gist_length} tokens: {text}",
        qa_prompt_template: str = "Based on the following context, answer the question. Context: {context} Question: {question}",
        lookup_prompt_template: str = "Based on the following summaries of pages and the question, which page(s) likely contain the details needed? Question: {question} Summaries: {summaries}",
        **kwargs: Any,
    ):
        if config is None:
            config = {}
        config = {**config, **kwargs}

        super().__init__(
            chunker=chunker,
            embedding_fn=embedding_fn,
            preprocess_fn=preprocess_fn,
            config=config,
        )

        self.llm_provider = llm_provider  # Updated instance variable

        # Retrieve parameters from self.config, allowing them to be overridden by what was passed in 'config' or 'kwargs'
        self.episode_token_limit = self.config.get(
            "episode_token_limit", episode_token_limit
        )
        self.gist_length = self.config.get(
            "gist_length", gist_length
        )  # Used as max_new_tokens for gisting
        self.gist_prompt_template = self.config.get(
            "gist_prompt_template", gist_prompt_template
        )
        self.qa_prompt_template = self.config.get(
            "qa_prompt_template", qa_prompt_template
        )
        self.lookup_prompt_template = self.config.get(
            "lookup_prompt_template", lookup_prompt_template
        )

        # Configuration for model names and max_new_tokens
        self.gist_model_name = self.config.get("gist_model_name", "distilgpt2")
        self.lookup_model_name = self.config.get("lookup_model_name", "distilgpt2")
        self.qa_model_name = self.config.get("qa_model_name", "distilgpt2")

        self.lookup_max_tokens = self.config.get("lookup_max_tokens", 50)
        self.qa_max_new_tokens = self.config.get("qa_max_new_tokens", 250)

        # Removed warning for local_llm_pipeline

    def _paginate_episodes(self, text: str) -> List[str]:
        if not text:
            return []
        episodes = text.split("\n\n")
        episodes = [ep.strip() for ep in episodes if ep.strip()]
        return episodes if episodes else ([text.strip()] if text.strip() else [])

    def _generate_gist(self, episode_text: str) -> str:
        if not episode_text:
            return ""
        # Note: gist_length in the prompt is illustrative; the actual control is max_new_tokens.
        prompt = self.gist_prompt_template.format(
            gist_length=self.gist_length, text=episode_text
        )

        if self.llm_provider:
            try:
                # self.gist_length is used as max_new_tokens for the gisting call
                return self.llm_provider.generate_response(
                    prompt=prompt,
                    model_name=self.gist_model_name,
                    max_new_tokens=self.gist_length,
                )
            except Exception as e:
                print(f"Error calling llm_provider for gisting: {e}")
                return f"Error generating gist. Input: {episode_text[:50]}..."
        return f"Simulated gist for episode: {episode_text[:50]}..."

    def _select_relevant_episodes(
        self, question: str, episode_gists: List[Tuple[int, str]]
    ) -> List[int]:
        if not question or not episode_gists:
            return []

        formatted_summaries = []
        for index, gist_text in episode_gists:
            formatted_summaries.append(f"Page {index + 1}: {gist_text}")

        summaries_str = "\n".join(formatted_summaries)
        prompt = self.lookup_prompt_template.format(
            question=question, summaries=summaries_str
        )
        selected_indices: List[int] = []

        if self.llm_provider:
            try:
                response = self.llm_provider.generate_response(
                    prompt=prompt,
                    model_name=self.lookup_model_name,
                    max_new_tokens=self.lookup_max_tokens,
                )
                raw_indices = response.replace("Page", "").split(",")
                for idx_str in raw_indices:
                    idx_str = idx_str.strip()
                    if idx_str.isdigit():
                        selected_indices.append(int(idx_str) - 1)
            except Exception as e:
                print(
                    f"Error during LLM call or parsing in _select_relevant_episodes: {e}"
                )
        else:
            if episode_gists:
                print(
                    "Simulating LLM call in _select_relevant_episodes, selecting first episode if available."
                )
                selected_indices.append(0)

        valid_indices = {
            idx for idx in selected_indices if 0 <= idx < len(episode_gists)
        }
        return sorted(list(valid_indices))

    def compress(
        self,
        text_or_chunks: Union[str, List[str]],
        llm_token_budget: int,
        **kwargs: Any,
    ) -> CompressedMemory:
        start_time = time.monotonic()
        trace_steps = []
        query: Optional[str] = kwargs.get("query", None)

        input_text: str
        if isinstance(text_or_chunks, list):
            input_text = " ".join(text_or_chunks)
            trace_steps.append(
                {
                    "type": "input_processing",
                    "details": "Input converted from list of chunks to single string.",
                }
            )
        else:
            input_text = str(text_or_chunks)
            trace_steps.append(
                {"type": "input_processing", "details": "Input is a single string."}
            )

        episodes = self._paginate_episodes(input_text)
        trace_steps.append(
            {
                "type": "pagination",
                "details": f"Text divided into {len(episodes)} episodes.",
            }
        )

        if not episodes:
            processing_time_ms = (time.monotonic() - start_time) * 1000
            trace = CompressionTrace(
                engine_name=self.id,
                strategy_params=self.config,
                input_summary={
                    "original_length": len(input_text),
                    "budget": llm_token_budget,
                    "num_episodes": 0,
                    "query": query,
                },
                steps=trace_steps,
                output_summary={"compressed_length": 0, "text": ""},
                processing_ms=processing_time_ms,
            )
            cm = CompressedMemory(text="")
            cm.trace = trace
            cm.engine_id = self.id
            cm.engine_config = self.config
            return cm

        episode_gists_texts: List[str] = []
        for i, episode_text in enumerate(episodes):
            gist_start_time = time.monotonic()
            gist = self._generate_gist(episode_text)
            gist_processing_time_ms = (time.monotonic() - gist_start_time) * 1000
            episode_gists_texts.append(gist)
            trace_steps.append(
                {
                    "type": "gisting_episode",
                    "details": {
                        "episode_index": i,
                        "episode_preview": (
                            episode_text[:100] + "..."
                            if len(episode_text) > 100
                            else episode_text
                        ),
                        "gist_preview": gist[:100] + "..." if len(gist) > 100 else gist,
                        "original_episode_length": len(episode_text),
                        "gist_length": len(gist),  # length here is char length
                        "processing_ms": gist_processing_time_ms,
                    },
                }
            )

        final_text = ""

        if query:
            trace_steps.append(
                {
                    "type": "task_detection",
                    "details": f"Query detected: '{query[:100]}...'. Processing as QA.",
                }
            )
            indexed_gists_for_selection: List[Tuple[int, str]] = list(
                enumerate(episode_gists_texts)
            )

            selection_start_time = time.monotonic()
            selected_episode_indices = self._select_relevant_episodes(
                query, indexed_gists_for_selection
            )
            selection_processing_ms = (time.monotonic() - selection_start_time) * 1000
            trace_steps.append(
                {
                    "type": "episode_selection_for_qa",
                    "details": {
                        "question": query[:100] + "...",
                        "num_gists_considered": len(indexed_gists_for_selection),
                        "selected_indices": selected_episode_indices,
                        "gists_preview_of_selected": {
                            idx: episode_gists_texts[idx][:50] + "..."
                            for idx in selected_episode_indices
                        },
                        "processing_ms": selection_processing_ms,
                    },
                }
            )

            context_parts = []
            for i, original_episode_text in enumerate(episodes):
                if i in selected_episode_indices:
                    context_parts.append(original_episode_text)
                else:
                    if i < len(episode_gists_texts):
                        context_parts.append(episode_gists_texts[i])
            final_context = "\n\n---\n\n".join(context_parts)
            trace_steps.append(
                {
                    "type": "context_construction_for_qa",
                    "details": {
                        "num_parts": len(context_parts),
                        "context_preview": final_context[:200] + "...",
                        "chars_full_text_in_context": sum(
                            len(episodes[i])
                            for i in selected_episode_indices
                            if i < len(episodes)
                        ),
                        "chars_gists_in_context": sum(
                            len(episode_gists_texts[i])
                            for i in range(len(episodes))
                            if i not in selected_episode_indices
                            and i < len(episode_gists_texts)
                        ),
                    },
                }
            )

            qa_prompt = self.qa_prompt_template.format(
                context=final_context, question=query
            )
            qa_llm_start_time = time.monotonic()

            if self.llm_provider:
                try:
                    # Using self.qa_max_new_tokens for the QA LLM call.
                    answer = self.llm_provider.generate_response(
                        prompt=qa_prompt,
                        model_name=self.qa_model_name,
                        max_new_tokens=self.qa_max_new_tokens,
                    )
                    final_text = answer
                    qa_llm_processing_ms = (time.monotonic() - qa_llm_start_time) * 1000
                    trace_steps.append(
                        {
                            "type": "qa_llm_call",
                            "details": {
                                "prompt_preview": qa_prompt[:200] + "...",
                                "answer_preview": answer[:100] + "...",
                                "processing_ms": qa_llm_processing_ms,
                            },
                        }
                    )
                except Exception as e:
                    print(f"Error during QA LLM call with llm_provider: {e}")
                    final_text = "Error generating answer."
                    qa_llm_processing_ms = (time.monotonic() - qa_llm_start_time) * 1000
                    trace_steps.append(
                        {
                            "type": "qa_llm_call_error",
                            "details": {
                                "error": str(e),
                                "processing_ms": qa_llm_processing_ms,
                            },
                        }
                    )
            else:
                final_text = f"Simulated answer for query: {query[:50]}..."
                qa_llm_processing_ms = (time.monotonic() - qa_llm_start_time) * 1000
                trace_steps.append(
                    {
                        "type": "qa_llm_call",
                        "details": "LLM provider not available, using simulated answer.",
                        "processing_ms": qa_llm_processing_ms,
                    }
                )

            tokenizer = (
                kwargs.get("tokenizer") or _DEFAULT_TOKENIZER or (lambda t: t.split())
            )
            if token_count(tokenizer, final_text) > llm_token_budget:
                original_answer_tokens = token_count(tokenizer, final_text)
                final_text = truncate_text(tokenizer, final_text, llm_token_budget)
                trace_steps.append(
                    {
                        "type": "answer_truncation",
                        "details": (
                            f"Answer (tokens {original_answer_tokens}) exceeded budget ({llm_token_budget})."
                            f" Truncated to {token_count(tokenizer, final_text)}."
                        ),
                    }
                )
        else:
            trace_steps.append(
                {
                    "type": "task_detection",
                    "details": "No query detected. Processing as Summarization.",
                }
            )
            concatenated_gists = "\n\n---\n\n".join(episode_gists_texts)
            trace_steps.append(
                {
                    "type": "concatenation_for_summary",
                    "details": f"Generated {len(episode_gists_texts)} gists. Total length: {len(concatenated_gists)}.",
                }
            )
            final_text = concatenated_gists
            tokenizer = (
                kwargs.get("tokenizer") or _DEFAULT_TOKENIZER or (lambda t: t.split())
            )
            if token_count(tokenizer, concatenated_gists) > llm_token_budget:
                original_summary_tokens = token_count(tokenizer, concatenated_gists)
                final_text = truncate_text(
                    tokenizer, concatenated_gists, llm_token_budget
                )
                trace_steps.append(
                    {
                        "type": "summary_truncation",
                        "details": (
                            f"Concatenated gists (tokens {original_summary_tokens}) exceeded budget ({llm_token_budget})."
                            f" Truncated to {token_count(tokenizer, final_text)}."
                        ),
                    }
                )
            else:
                trace_steps.append(
                    {
                        "type": "budget_check",
                        "details": (
                            f"Concatenated gists (tokens {token_count(tokenizer, concatenated_gists)}) within budget ({llm_token_budget})."
                            " No truncation needed."
                        ),
                    }
                )

        processing_time_ms = (time.monotonic() - start_time) * 1000
        output_summary = {
            "compressed_length": len(final_text),
            "num_episodes": len(episodes),
            "num_gists": len(episode_gists_texts),
            "query_processed": bool(query),
            "final_text_preview": (
                final_text[:100] + "..." if len(final_text) > 100 else final_text
            ),
        }

        trace = CompressionTrace(
            engine_name=self.id,
            strategy_params=self.config,
            input_summary={
                "original_length": len(input_text),
                "budget": llm_token_budget,
                "query": query,
            },
            steps=trace_steps,
            output_summary=output_summary,
            processing_ms=processing_time_ms,
            final_compressed_object_preview=final_text[:100],
        )
        cm = CompressedMemory(
            text=final_text,
            metadata={
                "num_episodes": len(episodes),
                "num_gists": len(episode_gists_texts),
                "query_processed": bool(query),
                "final_answer_preview_if_qa": final_text[:100] if query else None,
            },
        )
        cm.trace = trace
        cm.engine_id = self.id # Ensure engine_id is set
        cm.engine_config = self.config # Ensure engine_config is set
        return cm

    def _compress_chunk(self, chunk_text: str) -> str:
        if not chunk_text.strip():
            return ""
        gist = self._generate_gist(chunk_text)
        return gist


# Self-register upon import
register_compression_engine(
    ReadAgentGistEngine.id,
    ReadAgentGistEngine,
    display_name="ReadAgent Gist Engine",
    source="experimental",
)
