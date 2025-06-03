from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

# Configuration and Data Models (from Step 1)
from .api_config import (
    CompactMemoryConfig,
    EmbeddingConfig,
    ChunkerConfig,
    LLMProviderAPIConfig,
    StrategyConfig,
    MemoryStoreConfig
)
from .api_models import (
    IngestionReport,
    CompressedMemoryContext,
    AgentInteractionResponse,
    SourceReference
)
# Custom Exceptions (from Step 2)
from .api_exceptions import (
    InitializationError,
    ConfigurationError,
    StrategyNotFoundError,
    LLMProviderError,
    TokenizerError
)

# Existing components (placeholders for now, will need actual imports)
# from .json_npy_store import JsonNpyVectorStore # Example
# from .embedding_pipeline import EmbeddingPipeline # Example
# from .chunker import Chunker, SentenceWindowChunker # Example
# from .token_utils import get_tokenizer # Example
# from .llm_providers_abc import LLMProvider # Example
# from .registry import get_compression_strategy_class # Example

# Real imports will replace some of these mocks below
from .json_npy_store import JsonNpyVectorStore
from .embedding_pipeline import EmbeddingPipeline # Assuming embed_text is part of this or not directly used by agent
from .chunker import Chunker, SentenceWindowChunker, FixedSizeChunker # Add relevant chunker types
from .model_utils import get_embedding_model_info # For embedding dimension
from .models import RawMemory # For ingestion
from .token_utils import get_tokenizer, Tokenizer # Assuming Tokenizer is a class/type alias
import uuid # For generating memory IDs if store doesn't do it.

# LLM Provider related imports
from .llm_providers_abc import LLMProvider # Abstract Base Class
from .llm_providers import ( # Assuming a structure like this
    OpenAIProvider,
    GeminiProvider,
    # LocalTransformersProvider, # Assuming this might be the name for a local provider
    # For now, let's assume a get_llm_provider helper or we'll inline a simple factory
    get_llm_provider # A helper function to get a provider instance
)

# Real Strategy Imports
from .compression.strategies_abc import CompressionStrategy # Real ABC
from .registry import get_compression_strategy_class # Real registry function


logger = logging.getLogger(__name__)

class CompactMemoryAgent:
    """
    The CompactMemoryAgent is the main stateful entry point for interacting with the Compact Memory system.

    It orchestrates text ingestion, chunking, embedding, storage, context retrieval,
    and context compression using configurable strategies. It can also coordinate
    with LLM providers to generate responses based on retrieved context.

    Key Responsibilities:
        - Ingesting textual information into a configured memory store.
        - Retrieving relevant context from memory based on a query.
        - Compressing the retrieved context using a specified strategy.
        - Processing user messages to retrieve context and optionally generate an LLM response.
    """
    def __init__(
        self,
        config: Union[Dict[str, Any], CompactMemoryConfig],
        default_strategy_id: Optional[str] = None,
        storage_path: Optional[Union[str, Path]] = None,
        # Optional direct injection for advanced use or testing
        llm_provider_override: Optional[BaseLLMProvider] = None,
        tokenizer_override: Optional[BaseTokenizer] = None,
        embedding_pipeline_override: Optional[EmbeddingPipeline] = None,
        memory_store_override: Optional[JsonNpyVectorStore] = None,
        chunker_override: Optional[BaseChunker] = None
    ):
        """
        Initializes the CompactMemoryAgent and its components.

        Based on the provided configuration, this constructor sets up the embedding pipeline,
        chunker, memory store, default LLM provider, default tokenizer, and all configured
        compression strategies.

        Args:
            config: The configuration for the agent. Can be a dictionary conforming to
                    `CompactMemoryConfig` schema or an instance of `CompactMemoryConfig`.
            default_strategy_id: The instance ID of the compression strategy to be used by default
                                 if no specific strategy is requested in `retrieve_context` or
                                 `process_message`. This ID must be one of the keys in the
                                 `strategies` dictionary within the `config`.
            storage_path: An optional path to the memory store. If provided, this overrides
                          any path specified in `config.memory_store_config.path`. Useful for
                          dynamically setting the store location.
            llm_provider_override: For advanced use or testing. Allows direct injection of an
                                   LLMProvider instance, bypassing configuration for the agent's
                                   default LLM provider.
            tokenizer_override: For advanced use or testing. Allows direct injection of a
                                Tokenizer instance, bypassing configuration for the agent's
                                default tokenizer.
            embedding_pipeline_override: For advanced use or testing. Allows direct injection
                                         of an EmbeddingPipeline instance.
            memory_store_override: For advanced use or testing. Allows direct injection of a
                                   memory store instance (e.g., JsonNpyVectorStore).
            chunker_override: For advanced use or testing. Allows direct injection of a
                              Chunker instance.

        Raises:
            ConfigurationError: If the provided `config` is invalid (e.g., wrong type,
                                does not parse correctly, or specifies an unsupported component type).
            InitializationError: If any essential component (embedding pipeline, store, chunker,
                                 or a configured strategy) fails to initialize. This can be due to
                                 issues like incorrect model names, missing API keys (for real components),
                                 or problems accessing resources.
            LLMProviderError: If the default LLM provider (if configured) fails to initialize.
            StrategyNotFoundError: If a configured strategy class ID is not found in the registry.
        """
        logger.info("Initializing CompactMemoryAgent...")

        if isinstance(config, dict):
            try:
                self.config = CompactMemoryConfig(**config)
                logger.info("Loaded configuration from dictionary.")
            except Exception as e:
                raise ConfigurationError(f"Failed to parse dictionary into CompactMemoryConfig: {e}", details=config)
        elif isinstance(config, CompactMemoryConfig):
            self.config = config
            logger.info("Using provided CompactMemoryConfig instance.")
        else:
            raise ConfigurationError("Invalid config type. Must be a dict or CompactMemoryConfig instance.")

        # Determine memory store path
        self.storage_path: Optional[Path] = None
        if storage_path:
            self.storage_path = Path(storage_path)
        elif self.config.memory_store_config.path:
            self.storage_path = Path(self.config.memory_store_config.path)

        if not self.storage_path and self.config.memory_store_config.type == "default_json_npy":
            raise InitializationError("Storage path must be provided for default_json_npy memory store if not in config.")
        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Memory storage path set to: {self.storage_path}")


        # 1. Initialize Embedding Pipeline
        if embedding_pipeline_override:
            self.embedding_pipeline = embedding_pipeline_override
            logger.info("Using overridden embedding pipeline.")
        else:
            try:
                # Real EmbeddingPipeline instantiation
                self.embedding_pipeline = EmbeddingPipeline(
                    model_name=self.config.default_embedding_config.model_name,
                    api_key=self.config.default_embedding_config.api_key, # Assuming it takes api_key directly
                    provider=self.config.default_embedding_config.provider,
                    options=self.config.default_embedding_config.options
                )
                # Get embedding dimension
                model_info = get_embedding_model_info(self.config.default_embedding_config.model_name)
                self.embedding_dimension = model_info.dimension
                logger.info(f"Embedding pipeline initialized with model: {self.config.default_embedding_config.model_name}, Dimension: {self.embedding_dimension}")
            except Exception as e:
                raise InitializationError(f"Failed to initialize embedding pipeline: {e}", details=self.config.default_embedding_config.model_dump())

        # 2. Initialize Memory Store
        if memory_store_override:
            self.store = memory_store_override
            logger.info("Using overridden memory store.")
        else:
            store_type = self.config.memory_store_config.type
            if store_type == "default_json_npy":
                if not self.storage_path:
                    raise InitializationError("Storage path is required for JsonNpyVectorStore.")
                try:
                    # Real JsonNpyVectorStore instantiation
                    self.store = JsonNpyVectorStore(
                        path=str(self.storage_path),
                        embedding_dim=self.embedding_dimension,
                        embedding_model_name=self.config.default_embedding_config.model_name,
                        **(self.config.memory_store_config.params or {})
                    )
                    self.store.load() # Load existing data
                    logger.info(f"JsonNpyVectorStore initialized and loaded from {self.storage_path}")
                except Exception as e:
                    raise InitializationError(f"Failed to initialize JsonNpyVectorStore: {e}", details=self.config.memory_store_config.model_dump())
            else:
                raise ConfigurationError(f"Unsupported memory store type: {store_type}")

        # 3. Initialize Chunker
        if chunker_override:
            self.chunker: Chunker = chunker_override # Added type hint for clarity
            logger.info("Using overridden chunker.")
        else:
            chunker_type = self.config.default_chunker_config.type
            chunker_params = self.config.default_chunker_config.params or {}
            try:
                if chunker_type == "sentence_window":
                    self.chunker = SentenceWindowChunker(**chunker_params)
                elif chunker_type == "fixed_size":
                    # Ensure tokenizer name is passed if FixedSizeChunker needs it
                    # self.tokenizer is still a BaseTokenizer (mock) at this stage.
                    # If FixedSizeChunker requires a specific real tokenizer instance not yet available,
                    # this might need adjustment later. For now, pass the name.
                    if 'tokenizer_name' not in chunker_params and self.config.default_tokenizer_name:
                        chunker_params['tokenizer_name'] = self.config.default_tokenizer_name
                        logger.info(f"Passing default_tokenizer_name '{self.config.default_tokenizer_name}' to FixedSizeChunker.")
                    elif 'tokenizer_name' in chunker_params:
                         logger.info(f"FixedSizeChunker using tokenizer_name from params: '{chunker_params['tokenizer_name']}'.")
                    else:
                        # This state might be problematic if FixedSizeChunker requires a tokenizer
                        logger.warning("FixedSizeChunker selected, but no tokenizer_name provided in params and no agent default_tokenizer_name configured.")
                    self.chunker = FixedSizeChunker(**chunker_params)
                else:
                    raise ConfigurationError(f"Unsupported chunker type: {chunker_type}")
                logger.info(f"Chunker initialized with type: {chunker_type}")
            except Exception as e:
                raise InitializationError(f"Failed to initialize chunker type '{chunker_type}': {e}", details=self.config.default_chunker_config.model_dump())

        # 4. Initialize Default LLM Provider (optional)
        self.llm_provider: Optional[LLMProvider] = None # Type hint updated
        if llm_provider_override:
            self.llm_provider = llm_provider_override # Assuming override is of correct type LLMProvider
            logger.info("Using overridden LLM provider.")
        elif self.config.default_llm_provider_config:
            try:
                # Use get_llm_provider helper or inline factory logic
                self.llm_provider = get_llm_provider(self.config.default_llm_provider_config)
                logger.info(f"Default LLM provider initialized: {self.config.default_llm_provider_config.provider} - {self.config.default_llm_provider_config.model_name}")
            except Exception as e:
                raise LLMProviderError(f"Failed to initialize default LLM provider '{self.config.default_llm_provider_config.provider}': {e}", details=self.config.default_llm_provider_config.model_dump())
        else:
            logger.info("No default LLM provider configured for the agent.")

        # 5. Initialize Default Tokenizer (optional, but good to have a default)
        self.tokenizer: Optional[Tokenizer] = None # Type hint updated
        self.agent_default_strategy_id = None # Will be set later if provided in __init__
        if tokenizer_override:
            self.tokenizer = tokenizer_override # Assuming override is of correct type Tokenizer
            logger.info("Using overridden tokenizer.")
        elif self.config.default_tokenizer_name:
            try:
                self.tokenizer = get_tokenizer(self.config.default_tokenizer_name)
                logger.info(f"Default tokenizer loaded: {self.config.default_tokenizer_name}")
            except Exception as e: # Broad exception, could be specific TokenizerError if defined by get_tokenizer
                logger.warning(f"Could not load default tokenizer '{self.config.default_tokenizer_name}': {e}. Some functionalities might be limited.")
                self.tokenizer = None
        else:
            logger.info("No default tokenizer name configured.")

        # 6. Load and prepare strategies
        self.strategies: Dict[str, CompressionStrategy] = {} # Type hint updated to real Strategy

        for strategy_instance_id, strategy_conf in self.config.strategies.items():
            try:
                # Get Strategy Class from registry
                strategy_class_id = strategy_conf.strategy_class_id
                actual_strategy_cls = get_compression_strategy_class(strategy_class_id)
                logger.info(f"Successfully retrieved strategy class '{actual_strategy_cls.__name__}' for id '{strategy_class_id}' from registry.")

                strategy_llm: Optional[LLMProvider] = self.llm_provider # Default to agent's LLM
                if strategy_conf.llm_config:
                    # Check if override matches this strategy's config to reuse the override instance
                    if llm_provider_override and hasattr(llm_provider_override, 'config') and \
                       strategy_conf.llm_config.provider == llm_provider_override.config.provider and \
                       strategy_conf.llm_config.model_name == llm_provider_override.config.model_name:
                        strategy_llm = llm_provider_override
                        logger.info(f"Strategy '{strategy_instance_id}' uses overridden LLM provider instance.")
                    else:
                        try:
                            strategy_llm = get_llm_provider(strategy_conf.llm_config)
                            logger.info(f"Strategy '{strategy_instance_id}' uses its own LLM: {strategy_conf.llm_config.provider} - {strategy_conf.llm_config.model_name}")
                        except Exception as e:
                            logger.warning(f"Failed to load LLM for strategy '{strategy_instance_id}' using config {strategy_conf.llm_config.model_dump()}. Error: {e}. Falling back to agent default.")
                            strategy_llm = self.llm_provider # Fallback
                elif self.llm_provider:
                    logger.info(f"Strategy '{strategy_instance_id}' uses agent's default LLM.")
                else:
                    logger.info(f"Strategy '{strategy_instance_id}' has no LLM configured (neither specific nor agent default).")

                strategy_tokenizer: Optional[Tokenizer] = self.tokenizer # Default to agent's tokenizer
                if strategy_conf.tokenizer_name:
                    if tokenizer_override and tokenizer_override.name == strategy_conf.tokenizer_name:
                        strategy_tokenizer = tokenizer_override
                        logger.info(f"Strategy '{strategy_instance_id}' uses overridden tokenizer instance.")
                    else:
                        try:
                            strategy_tokenizer = get_tokenizer(strategy_conf.tokenizer_name)
                            logger.info(f"Strategy '{strategy_instance_id}' uses its own tokenizer: {strategy_conf.tokenizer_name}")
                        except Exception as e:
                            logger.warning(f"Could not load tokenizer '{strategy_conf.tokenizer_name}' for strategy '{strategy_instance_id}': {e}. Falling back to agent default or None.")
                            strategy_tokenizer = self.tokenizer # Fallback
                elif self.tokenizer:
                    logger.info(f"Strategy '{strategy_instance_id}' uses agent's default tokenizer.")
                else:
                    logger.info(f"Strategy '{strategy_instance_id}' has no tokenizer configured (neither specific nor agent default).")

                # The real strategy class constructor might take llm_provider and tokenizer directly.
                # It might also take the whole strategy_config (strategy_conf).
                # Assuming it takes specific params, llm, and tokenizer for now.
                strategy_specific_params = strategy_conf.params if strategy_conf.params is not None else {}

                self.strategies[strategy_instance_id] = actual_strategy_cls(
                    params=strategy_specific_params,
                    llm_provider=strategy_llm,
                    tokenizer=strategy_tokenizer
                )
                logger.info(f"Successfully instantiated strategy '{strategy_instance_id}' of type '{actual_strategy_cls.__name__}'")
            except StrategyNotFoundError as e: # Catch specific error from registry
                raise InitializationError(f"Could not find strategy class for id '{strategy_conf.strategy_class_id}': {e}", details=strategy_conf.model_dump())
            except Exception as e: # Catch other errors during instantiation
                raise InitializationError(f"Failed to instantiate strategy '{strategy_instance_id}' of type '{strategy_conf.strategy_class_id}': {e}", details=strategy_conf.model_dump())

        logger.info(f"Agent initialized with {len(self.strategies)} strategies.")

        self.agent_default_strategy_id = default_strategy_id
        if self.agent_default_strategy_id and self.agent_default_strategy_id not in self.config.strategies:
            logger.warning(f"Default strategy_id '{self.agent_default_strategy_id}' provided but not found in strategy configurations. Will fallback if needed.")

    def ingest(self, text: str, metadata: Optional[Dict[str, Any]] = None, user_id: Optional[str] = None) -> IngestionReport:
        """
        Ingests text into the agent's memory store.

        The process involves:
        1. Chunking the input text using the configured chunker.
        2. Generating embeddings for each chunk using the configured embedding pipeline.
        3. Storing the chunks, their embeddings, and associated metadata (including the
           provided metadata and user_id) as RawMemory objects in the memory store.

        Args:
            text: The text content to ingest.
            metadata: Optional dictionary of metadata to associate with the ingested text
                      and all its derived chunks. Example: `{"doc_id": "mydoc_01", "source_url": "..."}`.
            user_id: Optional identifier for the user associated with this text. If provided,
                     it will be stored in the metadata of each chunk, allowing for
                     user-specific retrieval later.

        Returns:
            An IngestionReport detailing the outcome of the ingestion process, including
            status, number of items processed/failed, and item IDs.

        Raises:
            IngestionError: If a critical error occurs during any part of the ingestion
                            process (e.g., embedding failure, storage failure) that prevents
                            successful completion. Some partial failures might be reported in
                            the IngestionReport without raising an exception.
        """
        logger.info(f"Ingesting text (first 50 chars): '{text[:50]}' with metadata: {metadata}, user_id: {user_id}")
        start_time = time.time()

        if not text:
            logger.warning("Ingestion called with empty text.")
            return IngestionReport(
                status="failure",
                message="Cannot ingest empty text.",
                items_processed=0,
                items_failed=1,
                processing_time_ms=(time.time() - start_time) * 1000
            )

        try:
            # 1. Chunk text
            # In a real scenario, chunker might return rich chunk objects.
            # For now, assume it returns a list of strings.
            chunks: List[str] = self.chunker.chunk_text(text)
            if not chunks:
                logger.info("Chunking resulted in no chunks.")
                return IngestionReport(
                    status="success", # Or "failure" depending on desired behavior for no chunks
                    message="Text resulted in no processable chunks.",
                    items_processed=0,
                    items_failed=0, # Or 1 if this is considered a failure
                    processing_time_ms=(time.time() - start_time) * 1000
                )
            logger.info(f"Text chunked into {len(chunks)} chunks.")

            # 2. Generate embeddings for each chunk
            # This will be replaced with actual call to self.embedding_pipeline.embed_texts(chunks)
            chunk_embeddings: List[List[float]] = self.embedding_pipeline.embed_texts(chunks)
            logger.info(f"Generated {len(chunk_embeddings)} embeddings for chunks.")

            # 3. Prepare metadata for each chunk
            # If a single metadata dict is provided, apply it to all chunks.
            # User ID should also be associated with each chunk if provided.
            # The actual store.add() might expect a list of metadata dicts.

            # For JsonNpyStore, it expects List[RawMemory] or similar.
            # We need to adapt this part when using the real store.
            # For now, we'll assume the store's `add` method can handle this structure,
            # or we'll prepare a mock structure it expects.

            # Mocking what might be passed to a store like JsonNpyStore.add which takes RawMemory objects
            # This part will need significant change when integrating real store and models.py RawMemory
            processed_item_ids = []
            raw_memories_to_add = []
            for i, chunk_text_content in enumerate(chunks):
                chunk_meta = metadata.copy() if metadata else {}
                chunk_meta["original_text_hash"] = hash(text) # Example, could be a document ID
                chunk_meta["chunk_index"] = i
                if user_id:
                    chunk_meta["user_id"] = user_id

                # This is a simplified representation. The actual RawMemory or equivalent
                # would be created here. For the mock store, we might not need full RawMemory.
                # Let's assume our mock store's `add` method can take these dicts directly for now.
                memory_item = {
                    "text": chunk_text_content,
                    "embedding": chunk_embeddings[i],
                    "metadata": chunk_meta,
                    # "id": self.store.generate_id() # Store might generate IDs
                }
                raw_memories_to_add.append(memory_item)
                # In a real scenario, IDs might come from the store after adding.
                processed_item_ids.append(f"mock_chunk_id_{i}")


            # 4. Store the original text, its embedding, and metadata into the memory store
            # This will be replaced with actual call to self.store.add(...)
            # The current mock JsonNpyStore.add is a pass-through.
            # Let's assume it returns a list of IDs or some status.
            try:
                    # Real store.add expects List[RawMemory]
                    self.store.add(raw_memories_to_add)
                    logger.info(f"Successfully added {len(raw_memories_to_add)} items to JsonNpyVectorStore.")
            except Exception as e:
                    logger.error(f"Error during store.add operation with JsonNpyVectorStore: {e}", exc_info=True)
                    raise IngestionError(f"Failed to store RawMemory objects: {e}")

            processing_time_ms = (time.time() - start_time) * 1000
            logger.info(f"Ingestion completed in {processing_time_ms:.2f} ms.")

            return IngestionReport(
                status="success",
                message=f"Successfully ingested {len(chunks)} chunks.",
                items_processed=len(chunks),
                items_failed=0,
                item_ids=processed_item_ids, # Actual IDs from RawMemory objects
                processing_time_ms=processing_time_ms
            )

        except InitializationError: # Should not happen here if agent is initialized
            raise
        except ConfigurationError: # Should not happen here
            raise
        except IngestionError as e:
            logger.error(f"Ingestion failed: {e}", exc_info=True)
            processing_time_ms = (time.time() - start_time) * 1000
            return IngestionReport(
                status="failure",
                message=str(e),
                items_processed=0, # Or count partially processed items if applicable
                items_failed=1, # Assuming one text input failed. If multiple chunks, this could be len(chunks)
                processing_time_ms=processing_time_ms
            )
        except Exception as e:
            logger.error(f"An unexpected error occurred during ingestion: {e}", exc_info=True)
            processing_time_ms = (time.time() - start_time) * 1000
            return IngestionReport(
                status="failure",
                message=f"An unexpected error occurred: {e}",
                items_processed=0,
                items_failed=1,
                processing_time_ms=processing_time_ms
            )

    def retrieve_context(
        self,
        query: str,
        strategy_id: Optional[str] = None, # Instance ID of the strategy
        budget: Optional[int] = None, # e.g. token budget
        user_id: Optional[str] = None,
        # strategy_kwargs: Optional[Dict[str, Any]] = None # Alternative way to pass strategy_kwargs
        **strategy_kwargs # Allows direct keyword arguments for the strategy
    ) -> CompressedMemoryContext:
        """
        Retrieves relevant context from memory and compresses it using a specified strategy.

        The process involves:
        1. Embedding the input query.
        2. Querying the memory store for relevant raw memories/chunks using the query embedding.
           User ID can be used for filtering if the store supports it.
        3. Selecting a compression strategy: uses `strategy_id` if provided, otherwise falls
           back to the agent's `default_strategy_id`, or the first available strategy.
        4. Invoking the selected strategy's `compress` method with the retrieved memories,
           the specified budget, and other contextual information (query, tokenizer, LLM provider).

        Args:
            query: The input query string for which to retrieve and compress context.
            strategy_id: The instance ID of the compression strategy to use (must be one of the
                         keys in the `strategies` dictionary of the agent's configuration).
                         If None, the agent's `default_strategy_id` is used.
            budget: An optional budget constraint for the compression strategy (e.g., token limit,
                    number of items). The interpretation of the budget is strategy-dependent.
            user_id: Optional identifier for the user. This can be used by the memory store
                     to filter retrieval to only memories associated with this user.
            **strategy_kwargs: Additional keyword arguments that will be passed down to the
                               selected strategy's `compress` method. This allows for passing
                               strategy-specific parameters at runtime (e.g., `top_k_retrieval`
                               for store querying, or dynamic parameters for the strategy itself).

        Returns:
            A CompressedMemoryContext object containing the compressed text, source references,
            and metadata about the compression process.

        Raises:
            RetrievalError: If the query is empty, or if an unexpected error occurs during
                            memory retrieval or context compression.
            StrategyNotFoundError: If the specified `strategy_id` (or the agent's default)
                                   cannot be found or if no strategies are configured.
        """
        logger.info(f"Retrieving context for query (first 50 chars): '{query[:50]}', strategy_id: {strategy_id}, budget: {budget}, user_id: {user_id}, strategy_kwargs: {strategy_kwargs}")
        start_time = time.time()

        if not query:
            logger.warning("Retrieve context called with empty query.")
            # Or raise ValueError("Query cannot be empty") - for now, return mock error context
            # This behavior might need refinement: what should an empty query return?
            # For now, let's assume it's an error or returns an empty context.
            # Raising an exception might be cleaner for an API.
            raise RetrievalError("Query cannot be empty.")

        try:
            # 1. Embed the input query
            # This will be replaced with actual call to self.embedding_pipeline.embed_query(query)
            query_embedding: List[float] = self.embedding_pipeline.embed_query(query)
            logger.info(f"Query embedded successfully.")

            # 2. Retrieve relevant raw memories/chunks from the memory store
            # This will be replaced with actual call to self.store.query(...)
            # The store might return a list of RawMemory objects or similar.
            # For now, assume it returns a list of simple objects/dicts that strategies can handle.
            # Filter by user_id if applicable (mock store query needs to support this)

            # Our mock store's query is basic. Let's simulate what it might return.
            # In a real scenario, store.query would take query_embedding, top_k, user_id etc.
            # For now, the mock store query returns `[]`. We need to make it return some mock data for the strategy.

            # Let's define that our mock store's query will just return a list of mock text chunks.
            # To make this testable without changing the store's mock query method now,
            # we will prepare some mock data directly here.
            # In a real implementation, this data comes from `self.store.query(...)`
            raw_memory_objects_from_store = self.store.query(
                query_embedding=query_embedding,
                user_id=user_id,
                top_k=strategy_kwargs.get('top_k_retrieval', 10) # Allow top_k to be passed in kwargs
            )
            logger.info(f"Retrieved {len(raw_memory_objects_from_store)} raw memory objects from store.")

            # The strategies will receive these raw_memory_objects.
            # If strategies expect simple text, we might need an adaptation step here,
            # but ideally, strategies are designed to work with RawMemory or similar rich objects.
            raw_memories_for_strategy = raw_memory_objects_from_store

            # 3. Select the compression strategy
            active_strategy_instance_id: Optional[str] = strategy_id
            if not active_strategy_instance_id:
                if hasattr(self, 'agent_default_strategy_id') and self.agent_default_strategy_id:
                    active_strategy_instance_id = self.agent_default_strategy_id
                    logger.info(f"No specific strategy_id provided, using agent's default: {active_strategy_instance_id}")
                elif self.strategies: # Fallback to first available if no agent default set
                    active_strategy_instance_id = next(iter(self.strategies.keys()))
                    logger.warning(f"No specific strategy_id and no agent default strategy set. Using first available: {active_strategy_instance_id}")
                else:
                    raise StrategyNotFoundError("No strategies configured for this agent, and no default specified.")

            if not active_strategy_instance_id or active_strategy_instance_id not in self.strategies:
                raise StrategyNotFoundError(f"Strategy with instance ID '{active_strategy_instance_id}' not found or not configured for this agent.")

            strategy_instance = self.strategies[active_strategy_instance_id]
            logger.info(f"Selected strategy: {active_strategy_instance_id} (Class: {strategy_instance.__class__.__name__})")

            # 4. Instantiate and invoke the strategy's compress method
            compression_kwargs = {
                "tokenizer": strategy_instance.tokenizer,
                "llm_provider": strategy_instance.llm_provider,
                "query_text": query,
                "query_embedding": query_embedding,
                **strategy_kwargs
            }

            compressed_context: CompressedMemoryContext = strategy_instance.compress(
                memories=raw_memories_for_strategy,
                budget=budget if budget is not None else -1,
                **compression_kwargs
            )

            processing_time_ms = (time.time() - start_time) * 1000
            if compressed_context.processing_time_ms is None:
               compressed_context.processing_time_ms = processing_time_ms
            else:
               current_total_time = (time.time() - start_time) * 1000
               # This assumes strategy_processing_time_ms was only for strategy internal work
               # The new time will be total time taken at agent level.
               compressed_context.processing_time_ms = current_total_time


            logger.info(f"Context retrieval and compression completed in {compressed_context.processing_time_ms:.2f} ms using strategy '{active_strategy_instance_id}'.")
            return compressed_context

        except StrategyNotFoundError as e:
            logger.error(f"Strategy not found: {e}", exc_info=True)
            raise
        except RetrievalError as e:
            logger.error(f"Retrieval failed: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during context retrieval: {e}", exc_info=True)
            raise RetrievalError(f"An unexpected error occurred during context retrieval: {e}", details={"original_exception": str(e)})

    def process_message(
        self,
        message: str,
        user_id: str, # Assuming user_id is mandatory
        session_id: Optional[str] = None,
        generate_response: bool = True,
        ingest_message_flag: bool = False, # Renamed to avoid clash with method name
        retrieval_strategy_id: Optional[str] = None,
        retrieval_budget: Optional[int] = None,
        # llm_generation_kwargs: Optional[Dict[str, Any]] = None # Alternative way
        **kwargs # For additional strategy or LLM generation kwargs
    ) -> AgentInteractionResponse:
        """
        Processes an incoming message, retrieves context, and optionally generates an LLM response.

        This is the primary method for user interaction with the agent. It orchestrates:
        1. Optional ingestion of the user's message into memory.
        2. Retrieval of relevant context using `retrieve_context` (which involves query embedding,
           store query, and strategy-based compression).
        3. Optional generation of a textual response using the agent's default LLM provider,
           based on the retrieved context and the input message.

        Args:
            message: The user's input message string.
            user_id: Identifier for the user making the request. Used for memory filtering
                     during retrieval and can be associated with ingested messages.
            session_id: Optional identifier for the current conversation session. Can be used
                        for session-specific logging or context management in the future.
            generate_response: If True, the agent will attempt to generate a response using its
                               default LLM provider after retrieving context. If False, no LLM
                               response is generated.
            ingest_message_flag: If True, the user's `message` will also be ingested into the
                                 memory store. Metadata including `source: "user_message_turn"`
                                 and `session_id` (if provided) will be added.
            retrieval_strategy_id: Optional instance ID of the compression strategy to use for
                                   context retrieval. If None, the agent's default strategy is used.
            retrieval_budget: Optional budget constraint for the context retrieval and compression
                              phase, passed to `retrieve_context`.
            **kwargs: Additional keyword arguments that can be passed down. These might include
                      parameters for `retrieve_context` (like `top_k_retrieval`) or parameters
                      for LLM response generation if `generate_response` is True.

        Returns:
            An AgentInteractionResponse object containing the LLM's response (if any),
            the context used, session/turn identifiers, and any error messages.

        Raises:
            This method aims to catch most exceptions from underlying operations (retrieval,
            LLM generation) and include them in the `AgentInteractionResponse.error_message`
            field. However, critical configuration or initialization errors might still propagate.
            Potential caught errors include `StrategyNotFoundError`, `RetrievalError`,
            `LLMProviderError`, `TokenizerError`.
        """
        logger.info(
            f"Processing message (first 50 chars): '{message[:50]}', user_id: {user_id}, session_id: {session_id}, "
            f"generate_response: {generate_response}, ingest_message: {ingest_message_flag}, "
            f"retrieval_strategy_id: {retrieval_strategy_id}, retrieval_budget: {retrieval_budget}, other kwargs: {kwargs}"
        )
        start_time = time.time()
        turn_id = f"turn_{int(time.time() * 1000)}" # Simple turn ID

        if not message:
            # Consider if this should raise an error or return a specific response
            logger.warning("Process message called with empty message.")
            # For now, let's create a context that reflects no query was made.
            empty_context = CompressedMemoryContext(
                compressed_text="",
                source_references=[],
                strategy_id_used="none",
                budget_info={"message": "No query message provided"},
                processing_time_ms=0.0
            )
            return AgentInteractionResponse(
                llm_response=None,
                context_used=empty_context,
                session_id=session_id,
                turn_id=turn_id,
                error_message="Empty message provided.",
                processing_time_ms=(time.time() - start_time) * 1000
            )

        try:
            # 1. Optionally, ingest the incoming message
            if ingest_message_flag:
                logger.info(f"Ingesting incoming message for user_id: {user_id}, session_id: {session_id}")
                ingest_meta = {"source": "user_message_turn"}
                if session_id:
                    ingest_meta["session_id"] = session_id

                # Call self.ingest. Errors during this ingestion should be caught
                # and potentially logged but might not always halt the process_message flow.
                try:
                    ingestion_report = self.ingest(text=message, metadata=ingest_meta, user_id=user_id)
                    if ingestion_report.status != "success":
                        logger.warning(f"Ingestion of user message had issues: {ingestion_report.message}")
                    else:
                        logger.info(f"User message ingested. Items processed: {ingestion_report.items_processed}")
                except Exception as e:
                    logger.error(f"Exception during message ingestion: {e}", exc_info=True)
                    # Decide if this is critical enough to stop processing the message.
                    # For now, we'll log and continue to retrieval & response generation.

            # 2. Retrieve relevant context
            # The query for retrieval might be the message itself, or a modified version
            # incorporating conversation history associated with session_id (complex, for later).
            # For now, the message is the query.

            # Pop strategy-specific kwargs from general kwargs if necessary,
            # or pass all kwargs and let retrieve_context/strategies handle them.
            # For now, pass all **kwargs through.
            retrieved_context: CompressedMemoryContext = self.retrieve_context(
                query=message,
                strategy_id=retrieval_strategy_id,
                budget=retrieval_budget,
                user_id=user_id,
                **kwargs # Pass through other relevant kwargs for strategy
            )
            logger.info(f"Context retrieved using strategy: {retrieved_context.strategy_id_used}")

            # 3. If generate_response is True and a default LLM provider is configured:
            llm_response_text: Optional[str] = None
            if generate_response:
                if not self.llm_provider:
                    logger.warning("Cannot generate response: No default LLM provider configured for the agent.")
                    # Not raising an error here, just returning no response, but an error_message in AgentInteractionResponse
                    final_processing_time_ms = (time.time() - start_time) * 1000
                    return AgentInteractionResponse(
                        llm_response=None,
                        context_used=retrieved_context,
                        session_id=session_id,
                        turn_id=turn_id,
                        error_message="LLM response generation skipped: No default LLM provider configured.",
                        processing_time_ms=final_processing_time_ms
                    )

                # Tokenizer check - agent's default tokenizer
                if not self.tokenizer and not getattr(self.llm_provider, "has_internal_tokenizer", False):
                    logger.warning("Cannot generate response: No default tokenizer configured for the agent, and LLM provider might require one.")
                    final_processing_time_ms = (time.time() - start_time) * 1000
                    return AgentInteractionResponse(
                        llm_response=None,
                        context_used=retrieved_context,
                        session_id=session_id,
                        turn_id=turn_id,
                        error_message="LLM response generation skipped: Tokenizer not available.",
                        processing_time_ms=final_processing_time_ms
                    )

                # Construct a prompt using the compressed_text from the retrieved context and the current message.
                # This is a very basic prompt template. Real applications would use more sophisticated templating
                # and potentially include conversation history based on session_id.
                prompt = f"Relevant Information:\n{retrieved_context.compressed_text}\n\nUser Message:\n{message}\n\nAssistant Response:"
                logger.debug(f"Constructed prompt for LLM (first 100 chars): {prompt[:100]}")

                # Use the agent's default LLM provider.
                # Generation kwargs can come from agent's config or be overridden by `**kwargs`.
                # For simplicity, let's assume `kwargs` passed to `process_message` can contain LLM generation args.
                # A more structured way would be to have an `llm_generation_kwargs` parameter.

                # Filter kwargs to only pass relevant generation parameters if needed, or let the provider handle it.
                # Example: generation_params = {k: v for k, v in kwargs.items() if k in RECOGNIZED_LLM_PARAMS}
                # For now, our mock LLM provider's generate_response is simple.

                try:
                    llm_response_text = self.llm_provider.generate_response(prompt, **kwargs) # Pass kwargs
                    logger.info(f"LLM response generated for user {user_id}, session {session_id}.")
                except Exception as e:
                    logger.error(f"Error during LLM response generation: {e}", exc_info=True)
                    # Not raising LLMProviderError here, but setting error_message in response
                    final_processing_time_ms = (time.time() - start_time) * 1000
                    return AgentInteractionResponse(
                        llm_response=None,
                        context_used=retrieved_context,
                        session_id=session_id,
                        turn_id=turn_id,
                        error_message=f"LLM response generation failed: {e}",
                        processing_time_ms=final_processing_time_ms
                    )
            else:
                logger.info("Response generation skipped as per `generate_response=False`.")

            # 4. Package the LLM's response, context, and other details into AgentInteractionResponse
            final_processing_time_ms = (time.time() - start_time) * 1000
            logger.info(f"Message processing completed in {final_processing_time_ms:.2f} ms.")

            return AgentInteractionResponse(
                llm_response=llm_response_text,
                context_used=retrieved_context,
                session_id=session_id,
                turn_id=turn_id, # Add turn identifier
                error_message=None, # Assuming overall success if we reach here
                processing_time_ms=final_processing_time_ms
            )

        except (StrategyNotFoundError, RetrievalError, LLMProviderError, TokenizerError, ConfigurationError, InitializationError) as e:
            # These are expected API errors, already logged by their respective operations or init
            logger.error(f"API error during message processing: {e}", exc_info=True)
            # Create a minimal context if retrieval failed before it could be formed
            failed_context = getattr(e, 'context_used', None) or CompressedMemoryContext(
                 compressed_text="", source_references=[], strategy_id_used="error",
                 budget_info={"error": str(e)}, processing_time_ms=0.0
             )
            if hasattr(e, 'context_used') and e.context_used: # type: ignore
                failed_context = e.context_used # type: ignore
            else: # Construct a minimal context if retrieval itself failed.
                 failed_context = CompressedMemoryContext(
                     compressed_text="", source_references=[], strategy_id_used="none_due_to_error",
                     budget_info={"error": "Context retrieval failed"}, processing_time_ms=0
                 )

            return AgentInteractionResponse(
                llm_response=None,
                context_used=failed_context, # Provide a valid, possibly empty, context
                session_id=session_id,
                turn_id=turn_id,
                error_message=str(e),
                processing_time_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            logger.error(f"An unexpected error occurred during message processing: {e}", exc_info=True)
            # Construct a minimal context for unexpected errors.
            unexpected_error_context = CompressedMemoryContext(
                compressed_text="", source_references=[], strategy_id_used="none_due_to_unexpected_error",
                budget_info={"error": "Unexpected error during processing"}, processing_time_ms=0
            )
            return AgentInteractionResponse(
                llm_response=None,
                context_used=unexpected_error_context,
                session_id=session_id,
                turn_id=turn_id,
                error_message=f"An unexpected error occurred: {e}",
                processing_time_ms=(time.time() - start_time) * 1000
            )
