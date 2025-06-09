from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)  # Added Tuple and Union
import json
import os
import uuid
import inspect
import time
import sys  # For printing warnings
from pathlib import Path  # Added import
import importlib
import logging  # Added for logging
import cloudpickle

import numpy as np
from pydantic import BaseModel  # For isinstance check in load

from ..chunker import Chunker, SentenceWindowChunker, FixedSizeChunker

# Ensure EngineConfig is imported for type hinting and instantiation
from ..engine_config import EngineConfig
from ..embedding_pipeline import (
    embed_text,
    get_embedding_dim,
)  # get_embedding_dim is imported here
from ..utils import calculate_sha256
from ..models import BeliefPrototype, RawMemory

# Updated imports for exceptions
from ..exceptions import (
    EngineLoadError,
    EngineSaveError,
    ConfigurationError,
    EmbeddingDimensionMismatchError,
    EngineError,  # Added EngineError
)
from ..vector_store import (
    VectorStore,
    create_vector_store,
    # InMemoryVectorStore # No longer needed for isinstance check in load
)


@dataclass
class CompressedMemory:
    text: str
    metadata: Optional[Dict[str, Any]] = None
    engine_id: Optional[str] = None
    engine_config: Optional[Dict[str, Any]] = None
    trace: Optional[CompressionTrace] = None


@dataclass
class CompressionTrace:
    engine_name: str
    strategy_params: Dict[str, Any]
    input_summary: Dict[str, Any]
    steps: List[Dict[str, Any]] = field(default_factory=list)
    output_summary: Dict[str, Any] = field(default_factory=dict)
    processing_ms: float | None = None
    final_compressed_object_preview: Optional[str] = None
    original_tokens: int | None = None
    compressed_tokens: int | None = None

    def add_step(self, step_type: str, details: Dict[str, Any]) -> None:
        self.steps.append({"type": step_type, "details": details})


class BaseCompressionEngine:
    id = "base_truncate"
    # Note: The `embeddings` property, which previously offered direct access to a
    # consolidated numpy array of embeddings, has been removed. Responsibility for
    # storing, managing, and providing access to embeddings (if needed) now lies
    # entirely with the specific `VectorStore` implementation. The engine delegates
    # persistence of vector data, associated texts, and indices to the vector store.
    config: EngineConfig
    embedding_fn: Callable[[str | Sequence[str]], np.ndarray]
    preprocess_fn: Optional[Callable[[str], str]]
    _chunker: Chunker
    vector_store: VectorStore
    _embed_accepts_preprocess: bool
    memories: List[
        Dict[str, Any]
    ]  # Already defined in __init__ but good to declare here
    memory_hashes: Set[str]  # Already defined in __init__ but good to declare here

    def __init__(
        self,
        *,
        chunker: Chunker | None = None,
        embedding_fn: Callable[[str | Sequence[str]], np.ndarray] = embed_text,
        preprocess_fn: Callable[[str], str] | None = None,
        vector_store: VectorStore | None = None,
        config: Optional[EngineConfig | Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Initializes the BaseCompressionEngine.

        The engine can be configured either through a full `EngineConfig` object,
        a dictionary, or by providing individual components like chunker,
        embedding_fn, etc., as keyword arguments. Configuration precedence is:
        kwargs > individual component objects > dict config > EngineConfig object.

        If `config` is provided (either as an `EngineConfig` instance or a dict),
        it can include `embedding_fn_path` and `preprocess_fn_path`. If these
        paths are set, the engine will prioritize loading the functions from these
        paths.

        If `embedding_fn` or `preprocess_fn` are provided directly as callable
        arguments (and their respective paths are not in `config`), the engine
        will use these provided functions. Furthermore, if these functions are
        determined to be importable (e.g., defined in a discoverable module and
        not lambdas or dynamically generated), their module paths will be
        automatically stored in `self.config.embedding_fn_path` and
        `self.config.preprocess_fn_path`. This facilitates serialization,
        allowing these functions to be reloaded when the engine is saved and
        subsequently loaded. A warning is issued if a directly provided function
        is not importable and thus cannot be serialized by path.

        Args:
            chunker: Optional Chunker instance. If provided, its ID might be
                     recorded in the config.
            embedding_fn: Callable for text embedding. Defaults to a standard
                          `embed_text` function. If a custom function is given
                          and is importable, its path is stored in the config.
            preprocess_fn: Optional callable for text preprocessing. If a custom
                           function is given and is importable, its path is
                           stored in the config.
            vector_store: Optional VectorStore instance. If provided, its type and
                          embedding dimension might be recorded in the config.
            config: Optional `EngineConfig` object or dict containing
                    configuration settings. Values from this can be overridden
                    by direct keyword arguments or other `**kwargs`.
            **kwargs: Additional configuration parameters that can override
                      settings in `config` or provide them if `config` is None.
                      These are typically fields of the `EngineConfig` model.

        Raises:
            ConfigurationError: If critical configuration like embedding dimension
                                cannot be determined, or if specified `embedding_fn_path`
                                or `preprocess_fn_path` (in `config` or `kwargs`)
                                point to non-loadable functions when `load()` is eventually called
                                on an engine instance created with such config. Also raised if
                                vector store creation fails due to configuration.
            EmbeddingDimensionMismatchError: If `embedding_dim` in config conflicts
                                           with default model's dimension when `embedding_fn`
                                           is the default. (Note: current behavior might warn
                                           and override rather than strictly raising this in all cases).
        """
        processed_config_data: Dict[str, Any] = {}
        if isinstance(config, EngineConfig):
            processed_config_data = config.model_dump(exclude_unset=False)
            if config.model_extra:
                processed_config_data.update(config.model_extra)
        elif isinstance(config, dict):
            processed_config_data = config.copy()

        processed_config_data.update(kwargs)

        if "vector_store" not in processed_config_data:
            processed_config_data["vector_store"] = "in_memory"

        if isinstance(chunker, Chunker):
            # getattr can return Any, ensure it's a string for the config key
            chunker_id_val_from_obj: str = str(
                getattr(chunker, "id", type(chunker).__name__)
            )
            processed_config_data["chunker_id"] = chunker_id_val_from_obj
        elif (
            chunker is not None
        ):  # chunker here is likely a string if not a Chunker instance
            chunker_id_str: str = str(chunker)
            processed_config_data["chunker_id"] = chunker_id_str

        if isinstance(vector_store, VectorStore):
            vector_store_id_val_from_obj: str = str(
                getattr(vector_store, "id", type(vector_store).__name__)
            )
            processed_config_data["vector_store"] = vector_store_id_val_from_obj
            if (
                hasattr(vector_store, "embedding_dim")
                and vector_store.embedding_dim is not None
            ):
                # embedding_dim should be int or None, ensure it is.
                # Pydantic model EngineConfig will validate this.
                processed_config_data["embedding_dim"] = vector_store.embedding_dim
        elif vector_store is not None:  # vector_store here is likely a string
            vector_store_id_str: str = str(vector_store)
            processed_config_data["vector_store"] = vector_store_id_str

        self.config: EngineConfig = EngineConfig(**processed_config_data)
        logging.debug(
            f"BaseCompressionEngine '{self.id}' initialized with config: {self.config}"
        )

        # Initialize embedding_fn and preprocess_fn to defaults and flags for serialization
        self.embedding_fn: Callable[[str | Sequence[str]], np.ndarray] = embed_text
        self.preprocess_fn: Optional[Callable[[str], str]] = None
        self._pickle_embedding_fn = False
        self._pickle_preprocess_fn = False

        # Load embedding_fn from path if provided in config
        if self.config.embedding_fn_path:
            try:
                module_path, func_name = self.config.embedding_fn_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                self.embedding_fn = getattr(module, func_name)
            except (ImportError, AttributeError, ValueError) as e:
                logging.warning(
                    f"Could not load embedding_fn from path '{self.config.embedding_fn_path}': {e}. "
                    f"Using default embed_text."
                )
                self.embedding_fn = embed_text
        elif embedding_fn is not None:
            self.embedding_fn = embedding_fn
            if embedding_fn is not embed_text:
                try:
                    module_name = getattr(embedding_fn, "__module__", None)
                    func_name = getattr(embedding_fn, "__name__", None)
                    if (
                        module_name
                        and func_name
                        and not module_name.startswith("__main__")
                        and not module_name.startswith("__")
                    ):
                        self.config.embedding_fn_path = f"{module_name}.{func_name}"
                    else:
                        self._pickle_embedding_fn = True
                except Exception:
                    self._pickle_embedding_fn = True

        # Load preprocess_fn from path if provided in config
        if self.config.preprocess_fn_path:
            try:
                module_path, func_name = self.config.preprocess_fn_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                self.preprocess_fn = getattr(module, func_name)
            except (ImportError, AttributeError, ValueError) as e:
                logging.warning(
                    f"Could not load preprocess_fn from path '{self.config.preprocess_fn_path}': {e}. "
                    f"No preprocess_fn will be used."
                )
                self.preprocess_fn = None
        elif preprocess_fn is not None:
            self.preprocess_fn = preprocess_fn
            try:
                module_name = getattr(preprocess_fn, "__module__", None)
                func_name = getattr(preprocess_fn, "__name__", None)
                if (
                    module_name
                    and func_name
                    and not module_name.startswith("__main__")
                    and not module_name.startswith("__")
                ):
                    self.config.preprocess_fn_path = f"{module_name}.{func_name}"
                else:
                    self._pickle_preprocess_fn = True
            except Exception:
                self._pickle_preprocess_fn = True

        if self.config.embedding_dim is None:
            # If embedding_fn was loaded from path, it might not be embed_text, so we check its identity.
            # Or if a custom embedding_fn was provided programmatically.
            if self.embedding_fn is embed_text:  # Check for default function identity
                try:
                    default_dim = get_embedding_dim()
                    if self.config.embedding_dim is None:
                        self.config.embedding_dim = default_dim
                    elif self.config.embedding_dim != default_dim:
                        # This case should be rare if dim is None, but as a safeguard
                        logging.warning(
                            f"Configured embedding_dim {self.config.embedding_dim} "
                            f"differs from default model dim {default_dim}. Using configured."
                        )
                        # Or raise EmbeddingDimensionMismatchError if strict adherence to default model is expected when fn is default
                except Exception as e:
                    logging.error(f"Failed to get default embedding dimension: {e}")
                    if (
                        self.config.embedding_dim is None
                    ):  # Only raise if not set at all
                        raise ConfigurationError(
                            "Could not determine default embedding dimension."
                        ) from e
            # If a custom embedding function is used, embedding_dim should ideally be set in the config.
            # If not, vector store creation might fail later or use a default.
            elif self.config.embedding_dim is None:
                logging.warning(
                    "Using a custom embedding_fn but embedding_dim is not set in config. This might lead to issues."
                )

        if chunker:
            self._chunker = chunker
        else:
            chunker_id_to_create = self.config.chunker_id
            if chunker_id_to_create == "fixed_size":
                self._chunker = FixedSizeChunker()
            elif chunker_id_to_create == "sentence_window":
                self._chunker = SentenceWindowChunker()
            else:
                print(
                    f"Warning: Unknown chunker_id '{chunker_id_to_create}' in config. Defaulting to SentenceWindowChunker.",
                    file=sys.stderr,
                )
                self._chunker = SentenceWindowChunker()

        # self.embedding_fn and self.preprocess_fn are now set above
        # Re-check signature if embedding_fn was potentially changed from default
        sig = inspect.signature(self.embedding_fn)
        self._embed_accepts_preprocess = "preprocess_fn" in sig.parameters

        if vector_store:
            self.vector_store = vector_store
        else:
            if self.config.embedding_dim is None:
                raise ValueError(
                    "embedding_dim could not be resolved for vector store creation."
                )  # This should ideally be ConfigurationError
            try:
                self.vector_store = create_vector_store(
                    store_type=self.config.vector_store,
                    embedding_dim=self.config.embedding_dim,
                    path=self.config.vector_store_path,
                )
            except (
                Exception
            ) as e:  # Catch errors from create_vector_store (e.g. invalid type)
                logging.error(
                    f"Failed to create vector store of type '{self.config.vector_store}': {e}"
                )
                raise ConfigurationError(f"Failed to create vector store: {e}") from e

        self.memories: List[Dict[str, Any]] = []
        self.memory_hashes: Set[str] = set()

    def _compress_chunk(self, chunk_text: str) -> str:
        return chunk_text

    def _ensure_index(self) -> None:
        pass

    def _embed(self, text_or_texts: str | Sequence[str]) -> np.ndarray:
        if self._embed_accepts_preprocess:
            return self.embedding_fn(text_or_texts, preprocess_fn=self.preprocess_fn)  # type: ignore
        processed_input = text_or_texts
        if self.preprocess_fn is not None:
            if isinstance(text_or_texts, str):
                processed_input = self.preprocess_fn(text_or_texts)
            else:
                processed_input = [self.preprocess_fn(t) for t in text_or_texts]
        return self.embedding_fn(processed_input)

    def ingest(self, text: str) -> List[str]:
        """
        Processes and ingests text into the memory.

        The text is chunked, (optionally) compressed, and embedded.
        New, unique entries (based on content hash) are added to the engine's
        internal `self.memories` list (which is persisted as `entries.json`)
        and also passed to the `VectorStore` via its
        `add_texts_with_ids_and_vectors` method.

        Args:
            text: The raw text string to ingest.

        Returns:
            A list of unique IDs for the newly ingested entries.

        Raises:
            EngineError: If the underlying vector store fails to add the data.
        """
        logging.info(f"Ingesting text of {len(text)} chars into engine '{self.id}'.")
        raw_chunks = self.chunker.chunk(text)
        logging.debug(f"Produced {len(raw_chunks)} raw chunks.")
        if not raw_chunks:
            return []

        processed_chunks = [self._compress_chunk(chunk) for chunk in raw_chunks]

        vecs = self._embed(processed_chunks)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)

        ids: List[str] = []
        new_entries_for_vector_store: List[Tuple[str, str, np.ndarray]] = []
        skipped_duplicates = 0

        for processed_chunk_text, vec in zip(processed_chunks, vecs):
            chunk_hash = calculate_sha256(processed_chunk_text)
            if chunk_hash not in self.memory_hashes:
                mid = uuid.uuid4().hex
                self.memories.append({"id": mid, "text": processed_chunk_text})
                new_entries_for_vector_store.append(
                    (mid, processed_chunk_text, vec.astype(np.float32))
                )
                ids.append(mid)
                self.memory_hashes.add(chunk_hash)
            else:
                skipped_duplicates += 1

        if new_entries_for_vector_store:
            try:
                self.vector_store.add_texts_with_ids_and_vectors(
                    new_entries_for_vector_store
                )
            except Exception as e:  # Catch potential VectorStoreError
                logging.error(f"VectorStore failed to add texts: {e}")
                # Decide if this should be a critical error or if engine can continue
                # For now, let's assume it's critical enough to warrant an EngineError
                raise EngineError(
                    f"Failed to ingest data into vector store: {e}"
                ) from e

        logging.info(
            f"Ingested {len(ids)} new memories, skipped {skipped_duplicates} duplicates."
        )
        return ids

    def recall(self, query: str, *, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Recalls relevant memories for a given query.

        The query is embedded, and the `VectorStore` is searched for the
        nearest neighbors. Texts for the retrieved IDs are then fetched
        from the `VectorStore` using `get_texts_by_ids`.

        Args:
            query: The query string.
            top_k: The number of top results to retrieve.

        Returns:
            A list of dictionaries, each containing "id", "text", and "score".

        Raises:
            EngineError: If an error occurs during vector store operations (find_nearest, get_texts_by_ids).
        """
        logging.info(
            f"Recall query: '{query[:50]}...', top_k: {top_k} from engine '{self.id}'."
        )
        vs_count = self.vector_store.count()

        if vs_count == 0:
            logging.debug("Vector store is empty, recall returning no results.")
            return []

        qvec = self._embed(query).reshape(1, -1).astype(np.float32)
        logging.debug(f"Query vector shape: {qvec.shape}")

        k = min(top_k, vs_count)
        if k == 0:
            logging.debug("Effective k is 0, recall returning no results.")
            return []

        try:
            nearest_ids_scores = self.vector_store.find_nearest(qvec[0], k)
            if not nearest_ids_scores:
                logging.debug("VectorStore find_nearest returned no results.")
                return []

            recalled_ids = [item_id for item_id, score in nearest_ids_scores]
            recalled_texts_map = self.vector_store.get_texts_by_ids(recalled_ids)
        except Exception as e:  # Catch potential VectorStoreError
            logging.error(f"Error during vector store operation in recall: {e}")
            raise EngineError(f"Failed recall due to vector store error: {e}") from e

        results: List[Dict[str, Any]] = []
        for item_id, score in nearest_ids_scores:
            text = recalled_texts_map.get(item_id, "")
            results.append({"id": item_id, "text": text, "score": float(score)})

        logging.info(f"Recall found {len(results)} results.")
        return results

    def compress(
        self,
        text_or_chunks: Union[str, List[str]],
        budget: Optional[int],
        previous_compression_result: Optional[CompressedMemory] = None,
        *,
        tokenizer: Any = None,
        **kwargs: Any,
    ) -> CompressedMemory:  # Added tokenizer and kwargs
        """
        Compresses the input text (or list of text chunks) to meet the given budget.

        This base implementation performs simple truncation after joining chunks if a list is provided.
        If budget is None, it will effectively not truncate.
        Subclasses should override this method to implement more sophisticated compression logic.

        Trace generation is controlled by `self.config.enable_trace`. If False,
        the `trace` field in the returned `CompressedMemory` object will be None.

        Args:
            text_or_chunks: The input string or list of strings to compress.
            budget: The target maximum number of characters (for this base impl.)
                    or tokens (for more advanced impl.) for the compressed output.
                    If None, no truncation based on budget is performed by this base method.
            previous_compression_result: Output from a preceding engine in a
                                         pipeline, if applicable.
            tokenizer: (Optional) A tokenizer instance or callable. Subclasses may use this.
            **kwargs: Additional engine-specific parameters.

        Returns:
            A CompressedMemory object containing the compressed text and
            optionally a trace of the compression process.
        """
        input_text_str: str
        if isinstance(text_or_chunks, list):
            input_text_str = " ".join(text_or_chunks)  # Join chunks with space
        else:
            input_text_str = text_or_chunks

        # Basic truncation logic, common to all engines if no other logic is applied.
        # Subclasses will typically override this or parts of it.
        if budget is not None:
            truncated_text = input_text_str[:budget]
        else:
            truncated_text = input_text_str  # No truncation if budget is None

        if not self.config.enable_trace:
            return CompressedMemory(
                text=truncated_text,
                trace=None,
                engine_id=self.id,
                engine_config=self.config.model_dump(mode="json"),
            )

        # Proceed with trace generation if enabled
        start_time = time.monotonic()

        # For BaseCompressionEngine, the "compression" is simple truncation.
        # More sophisticated engines will have more steps here.
        trace = CompressionTrace(
            engine_name=self.id,
            strategy_params={"budget": budget, **kwargs},
            input_summary={"original_length": len(input_text_str)},
            steps=[],
            output_summary={"compressed_length": len(truncated_text)},
            final_compressed_object_preview=truncated_text[:50],
        )
        trace.add_step("truncate", {"final_length": len(truncated_text)})
        trace.processing_ms = (time.monotonic() - start_time) * 1000

        return CompressedMemory(
            text=truncated_text,
            trace=trace,
            engine_id=self.id,
            engine_config=self.config.model_dump(mode="json"),
        )

    def save(self, path: os.PathLike | str) -> None:
        """
        Saves the state of the compression engine.

        This method saves:
        1.  `entries.json`: A list of dictionaries (`{'id': ..., 'text': ...}`)
            representing the engine's processed items, managed in `self.memories`.
        2.  `engine_manifest.json`: Contains engine configuration and metadata.

        It then calls `self.vector_store.save()` to instruct the vector store
        to persist its own data (including embeddings, texts, and indices) into
        a subdirectory named `vector_store_data` within the given path.
        The engine no longer saves `embeddings.npy` directly.

        Args:
            path: The directory path where the engine's data will be saved.

        Raises:
            EngineSaveError: If any error occurs during the save process, including
                             issues with file operations or vector store saving.
        """
        p = Path(path)
        logging.info(f"Saving engine '{self.id}' to {p}")
        try:
            p.mkdir(parents=True, exist_ok=True)

            with open(p / "entries.json", "w", encoding="utf-8") as fh:
                json.dump(self.memories, fh)
            logging.debug(f"Saved entries.json to {p / 'entries.json'}")

            manifest = {
                "engine_id": self.id,
                "engine_class": f"{self.__class__.__module__}.{self.__class__.__name__}",
                "config": self.config.model_dump(mode="json"),
                "metadata": {
                    "num_memories": len(self.memories),
                    "embedding_dim": self.config.embedding_dim,
                },
            }
            with open(p / "engine_manifest.json", "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2)
            logging.debug(f"Saved engine_manifest.json to {p / 'engine_manifest.json'}")

            if self._pickle_embedding_fn:
                with open(p / "embedding_fn.pkl", "wb") as fh:
                    cloudpickle.dump(self.embedding_fn, fh)
            if self._pickle_preprocess_fn and self.preprocess_fn is not None:
                with open(p / "preprocess_fn.pkl", "wb") as fh:
                    cloudpickle.dump(self.preprocess_fn, fh)

            self.vector_store.save(str(p / "vector_store_data"))
            if hasattr(self.vector_store, "proto_vectors"):
                np.save(p / "embeddings.npy", self.vector_store.proto_vectors)
            logging.info(f"Engine '{self.id}' saved successfully to {p}")
        except Exception as e:
            logging.error(f"Failed to save engine '{self.id}' to {p}: {e}")
            raise EngineSaveError(f"Failed to save engine to {p}: {e}") from e

    def load(self, path: os.PathLike | str) -> None:
        """
        Loads the state of the compression engine.

        This method loads:
        1.  `engine_manifest.json`: Restores engine configuration and metadata.
            This includes re-initializing `embedding_fn` and `preprocess_fn`
            if their paths are specified in the config.
        2.  `entries.json`: Populates `self.memories` and `self.memory_hashes`.

        It then calls `self.vector_store.load()` to instruct the vector store
        to load its own data from the `vector_store_data` subdirectory.
        The engine no longer loads `embeddings.npy` directly nor does it
        repopulate `InMemoryVectorStore`; these responsibilities are now
        fully delegated to the vector store implementation.

        Args:
            path: The directory path from which to load the engine's data.

        Raises:
            EngineLoadError: If loading fails due to missing files (manifest, entries.json),
                             corrupted data (JSON decode errors), issues loading dynamic
                             functions (`embedding_fn`, `preprocess_fn`), or failures
                             in vector store loading.
            ConfigurationError: If critical configuration like embedding dimension cannot
                                be determined from the loaded manifest and existing config.
            EmbeddingDimensionMismatchError: If a loaded embedding dimension from manifest
                                           conflicts significantly with other settings (though
                                           current behavior might log a warning and attempt to resolve).
        """
        p = Path(path)
        manifest_path = p / "engine_manifest.json"
        logging.info(f"Loading engine '{self.id}' from {p}")
        try:
            if not manifest_path.exists():
                raise FileNotFoundError(f"Engine manifest not found at {manifest_path}")

            with open(manifest_path, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            logging.debug(f"Loaded engine_manifest.json from {manifest_path}")

            loaded_manifest_config_dict = manifest.get("config", {})
            if not isinstance(loaded_manifest_config_dict, dict):  # Basic validation
                raise EngineLoadError(
                    f"Invalid 'config' format in manifest: {manifest_path}"
                )

        except FileNotFoundError as e:
            logging.error(
                f"Engine load failed: Manifest or essential file not found at {p}. Error: {e}"
            )
            raise EngineLoadError(f"Manifest or essential file not found: {e}") from e
        except json.JSONDecodeError as e:
            logging.error(
                f"Engine load failed: Could not decode JSON from manifest at {manifest_path}. Error: {e}"
            )
            raise EngineLoadError(f"Manifest JSON decode error: {e}") from e
        except (
            Exception
        ) as e:  # Catch other initial load errors (e.g. permission issues)
            logging.error(
                f"Engine load failed during initial manifest read from {p}: {e}"
            )
            raise EngineLoadError(f"Failed to read manifest: {e}") from e

        # Create a new EngineConfig instance from loaded, ensuring defaults for new fields
        # and then update with any existing self.config values that weren't in manifest (e.g. runtime extras)
        # This merge logic might need refinement based on desired precedence.
        # For now, manifest config takes precedence for saved fields.
        current_config_extras = (
            self.config.model_extra if self.config and self.config.model_extra else {}
        )
        merged_load_config = {**current_config_extras, **loaded_manifest_config_dict}
        self.config = EngineConfig(**merged_load_config)

        # --- BEGIN MODIFICATION: Re-initialize functions based on loaded config ---
        # Default assignments, in case paths are not found or lead to errors

        # Paths for serialized callables if module paths are not available
        embed_pickle_path = p / "embedding_fn.pkl"
        preprocess_pickle_path = p / "preprocess_fn.pkl"

        # Load embedding_fn from path if provided in the newly loaded config
        if self.config.embedding_fn_path:
            try:
                module_path, func_name = self.config.embedding_fn_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                self.embedding_fn = getattr(module, func_name)
            except (ImportError, AttributeError, ValueError) as e:
                logging.error(
                    f"Failed to load embedding_fn from path '{self.config.embedding_fn_path}': {e}"
                )
                raise EngineLoadError(
                    f"Failed to load embedding_fn from path '{self.config.embedding_fn_path}': {e}"
                ) from e
        elif embed_pickle_path.exists():
            with open(embed_pickle_path, "rb") as fh:
                self.embedding_fn = cloudpickle.load(fh)
                self._pickle_embedding_fn = True
        # else: self.embedding_fn remains default from __init__

        # Load preprocess_fn from path if provided in the newly loaded config
        if self.config.preprocess_fn_path:
            try:
                module_path, func_name = self.config.preprocess_fn_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                self.preprocess_fn = getattr(module, func_name)
            except (ImportError, AttributeError, ValueError) as e:
                logging.error(
                    f"Failed to load preprocess_fn from path '{self.config.preprocess_fn_path}': {e}"
                )
                raise EngineLoadError(
                    f"Failed to load preprocess_fn from path '{self.config.preprocess_fn_path}': {e}"
                ) from e
        elif preprocess_pickle_path.exists():
            with open(preprocess_pickle_path, "rb") as fh:
                self.preprocess_fn = cloudpickle.load(fh)
                self._pickle_preprocess_fn = True
        # else: self.preprocess_fn remains default from __init__

        # Re-check signature for the potentially updated embedding_fn
        sig = inspect.signature(self.embedding_fn)
        self._embed_accepts_preprocess = "preprocess_fn" in sig.parameters
        # --- END MODIFICATION ---

        try:
            metadata = manifest.get("metadata", {})
            if not isinstance(metadata, dict):  # Basic validation
                raise EngineLoadError(
                    f"Invalid 'metadata' format in manifest: {manifest_path}"
                )
            metadata_embedding_dim = metadata.get("embedding_dim")

            authoritative_embedding_dim = metadata_embedding_dim
            if (
                authoritative_embedding_dim is None
            ):  # Try from current config if not in metadata
                authoritative_embedding_dim = self.config.embedding_dim

            if authoritative_embedding_dim is None:
                raise ConfigurationError(  # Changed from ValueError
                    "Cannot determine embedding dimension for loading vector store. "
                    "It must be present in the manifest's metadata or engine config."
                )

            if self.config.embedding_dim is None:
                self.config.embedding_dim = authoritative_embedding_dim
            elif self.config.embedding_dim != authoritative_embedding_dim:
                warning_msg = (
                    f"Warning: EngineConfig embedding_dim ({self.config.embedding_dim}) "
                    f"mismatches effective dimension ({authoritative_embedding_dim}) from manifest/embeddings. "
                    "Using effective dimension."
                )
                logging.warning(warning_msg)
                print(warning_msg, file=sys.stderr)
                # Consider raising EmbeddingDimensionMismatchError here if strict consistency is required.
                # For now, warning and override is kept.
                # raise EmbeddingDimensionMismatchError(
                #    f"Configured dim {self.config.embedding_dim} != manifest dim {authoritative_embedding_dim}"
                # )
                self.config.embedding_dim = authoritative_embedding_dim

            self.vector_store = create_vector_store(
                store_type=self.config.vector_store,
                embedding_dim=self.config.embedding_dim,
                path=self.config.vector_store_path,
            )
            self.vector_store.load(str(p / "vector_store_data"))
            logging.debug("Vector store loaded.")

            entries_path = p / "entries.json"
            if not entries_path.exists():
                raise FileNotFoundError(
                    f"Engine entries file not found at {entries_path}"
                )
            with open(entries_path, "r", encoding="utf-8") as fh:
                self.memories = json.load(fh)
            self.memory_hashes = {
                calculate_sha256(mem["text"]) for mem in self.memories
            }
            logging.debug(f"Loaded {len(self.memories)} entries from entries.json.")

            logging.info(f"Engine '{self.id}' loaded successfully from {p}")

        except FileNotFoundError as e:  # Catch issues like missing entries.json
            logging.error(
                f"Engine load failed: Essential file not found. Path: {p}. Error: {e}"
            )
            raise EngineLoadError(f"Essential file not found during load: {e}") from e
        except (
            ConfigurationError,
            EmbeddingDimensionMismatchError,
        ) as e:  # Propagate specific config errors
            logging.error(f"Engine load failed due to configuration issue: {e}")
            raise  # Re-raise as these are already specific enough
        except (
            Exception
        ) as e:  # Catch other errors during vector_store creation, load, or entries.json processing
            logging.error(f"Engine load failed during data loading from {p}: {e}")
            raise EngineLoadError(f"Failed to load engine data: {e}") from e

    def rebuild_index(self) -> None:
        """
        Requests the underlying vector store to rebuild its search index.

        This can be useful to ensure the index is up-to-date after manual
        data changes or if the index is suspected to be stale.
        For persistent stores, this may also re-persist the rebuilt index.
        """
        logging.info(
            f"Index rebuilding requested for vector store of type: {self.config.vector_store}"
        )
        self.vector_store.rebuild_index()
        logging.info("Index rebuilding completed for vector store.")

    # The `embeddings` property has been removed as of version [VERSION_TAG_AFTER_THIS_CHANGE].
    # The BaseCompressionEngine no longer directly manages or exposes a consolidated embedding matrix.
    # If you need access to embeddings, you should interact with the `vector_store` instance directly,
    # assuming its specific implementation provides a method to access them.
    # For example, some vector stores might offer a `get_all_vectors()` method.
    # Consider whether direct access to all embeddings is necessary for your use case,
    # or if higher-level operations via the vector store are more appropriate.
    #
    # Previous implementation (now removed):
    # @property
    # def embeddings(self) -> np.ndarray:
    #     if hasattr(self.vector_store, 'proto_vectors') and self.vector_store.proto_vectors is not None:
    #         return self.vector_store.proto_vectors # type: ignore
    #     dim = self.config.embedding_dim if self.config.embedding_dim is not None else 0
    #     if dim == 0 :
    #         dim = get_embedding_dim()
    #     return np.zeros((0, dim), dtype=np.float32)

    @property
    def chunker(self) -> Chunker:
        return self._chunker

    @chunker.setter
    def chunker(self, value: Chunker) -> None:
        if not isinstance(value, Chunker):
            raise TypeError("chunker must implement Chunker interface")
        self._chunker = value
        if isinstance(value, FixedSizeChunker):
            self.config.chunker_id = "fixed_size"
        elif isinstance(value, SentenceWindowChunker):
            self.config.chunker_id = "sentence_window"
        else:
            self.config.chunker_id = getattr(value, "id", type(value).__name__)


__all__ = [
    "BaseCompressionEngine",
    "CompressedMemory",
    "CompressionTrace",
    "EngineConfig",
]
