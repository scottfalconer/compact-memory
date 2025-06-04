import numpy as np
from typing import Union, Sequence, List, Tuple, Dict, Optional, Any
import os
import tempfile
import shutil
import logging

# Setup basic logging for the demo
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Assuming compact_memory codebase is structured to allow these imports
# Adjust these imports based on your actual project structure
try:
    from compact_memory.agent import Agent
    from compact_memory.vector_store import BaseVectorStore, InMemoryVectorStore
    from compact_memory.embedding_pipeline import EmbeddingFunction # For type hinting
    # Attempt to import optional stores for the demo
    try:
        from compact_memory.vector_stores.chroma_adapter import ChromaVectorStoreAdapter
        CHROMADB_INSTALLED = True
    except ImportError:
        CHROMADB_INSTALLED = False
        ChromaVectorStoreAdapter = None # Placeholder
        logging.info("ChromaDB adapter not found, Chroma demo will be skipped.")
    try:
        from compact_memory.vector_stores.faiss_adapter import FaissVectorStoreAdapter
        FAISS_INSTALLED = True
    except ImportError:
        FAISS_INSTALLED = False
        FaissVectorStoreAdapter = None # Placeholder
        logging.info("FAISS adapter not found, FAISS demo will be skipped.")

except ImportError as e:
    logging.error(f"Could not import necessary Compact Memory components: {e}")
    logging.error("Ensure the Compact Memory library is installed and accessible in your PYTHONPATH.")
    # Exit if core components are missing
    sys.exit(1)


# --- 1. Define a Custom Embedding Function ---
def my_simple_embedding_fn(text_input: Union[str, Sequence[str]]) -> np.ndarray:
    """
    Simple hash-based embedding for demonstration.
    In a real scenario, this would use a proper model (OpenAI, Cohere, custom SentenceTransformer).
    Output should be 1D for single string, 2D for sequence of strings.
    """
    dim = 128  # Our custom embedding dimension

    if isinstance(text_input, str):
        is_single_string = True
        texts = [text_input]
    else:
        is_single_string = False
        texts = list(text_input)

    embeddings = []
    for t_item in texts:
        if not t_item:  # Handle empty string case
            embeddings.append(np.zeros(dim, dtype=np.float32))
            continue

        # Create a hash and convert to a fixed-size vector
        # Use first `dim` bytes of the hash for simplicity.
        # This is a very naive embedding, purely for structural demo.
        hash_object = hashlib.sha256(t_item.encode('utf-8', 'ignore'))
        hash_bytes = hash_object.digest()

        # Take first dim bytes and interpret as float32
        # This is not a good way to generate embeddings, but makes it fixed-length
        # and deterministic for the demo.
        num_floats = dim
        float_bytes = num_floats * 4 # 4 bytes per float32

        # Pad or truncate hash_bytes to ensure it's float_bytes long
        if len(hash_bytes) < float_bytes:
            padded_hash_bytes = hash_bytes + b'\0' * (float_bytes - len(hash_bytes))
        else:
            padded_hash_bytes = hash_bytes[:float_bytes]

        # Convert bytes to numpy array of float32
        # This is still problematic as direct byte to float conversion can lead to NaNs/Infs
        # A safer (but still naive) approach:
        # Interpret bytes as uint8, scale to [0,1], then create structured vector
        seed_vector = np.frombuffer(padded_hash_bytes, dtype=np.uint8).astype(np.float32) / 255.0

        # Ensure the vector has the correct dimension by repeating/padding if necessary
        # This step ensures the output vector has exactly 'dim' float32 numbers.
        if len(seed_vector) < dim:
            full_vector = np.zeros(dim, dtype=np.float32)
            full_vector[:len(seed_vector)] = seed_vector
            # If seed_vector is very short, this will have many zeros.
            # Could tile/repeat seed_vector to fill up 'dim' more meaningfully.
            # Example: if dim=128, seed_vector len 32, tile it 4 times.
            if len(seed_vector) > 0:
                 num_tiles = (dim + len(seed_vector) -1) // len(seed_vector)
                 tiled_vector = np.tile(seed_vector, num_tiles)[:dim]
                 full_vector = tiled_vector
            else: # seed_vector was empty (e.g. from empty hash_bytes, though hash isn't empty)
                 full_vector = np.zeros(dim, dtype=np.float32)

        elif len(seed_vector) > dim:
            full_vector = seed_vector[:dim]
        else:
            full_vector = seed_vector

        norm = np.linalg.norm(full_vector)
        embeddings.append(full_vector / (norm if norm > 0 else 1e-9)) # Normalize

    result_array = np.array(embeddings, dtype=np.float32)

    if is_single_string:
        return result_array[0] # Return 1D array for single string input
    return result_array # Return 2D array for sequence of strings


custom_embedding_dimension = 128

print("--- Demo: Agent with InMemoryVectorStore and Custom Embeddings ---")
agent_custom_embed = Agent(
    embedding_fn=my_simple_embedding_fn,
    embedding_dim=custom_embedding_dimension,
)

agent_custom_embed.add_memory("The sky is blue.") # Simplified add_memory
agent_custom_embed.add_memory("Grass is green.")
agent_custom_embed.add_memory("The sun is bright.")

query_text = "What color is the sky?"
logging.info(f"Querying with custom embeddings: '{query_text}'")
response_custom = agent_custom_embed.query(query_text)
logging.info(f"Response: {response_custom}")
print("\n")

# Temporary directory for this demo run
temp_demo_dir = tempfile.mkdtemp(prefix="cm_demo_")
logging.info(f"Created temporary directory for demos: {temp_demo_dir}")

try:
    if CHROMADB_INSTALLED and ChromaVectorStoreAdapter:
        print("--- Demo: Agent with ChromaVectorStoreAdapter and Custom Embeddings ---")
        chroma_storage_path = os.path.join(temp_demo_dir, "chroma_db_storage")
        os.makedirs(chroma_storage_path, exist_ok=True)
        logging.info(f"Using ChromaDB storage path: {chroma_storage_path}")

        chroma_store = ChromaVectorStoreAdapter(
            collection_name="custom_demo_collection_py", # Unique name for Python demo
            path=chroma_storage_path
        )

        agent_chroma = Agent(
            vector_store=chroma_store,
            embedding_fn=my_simple_embedding_fn,
            embedding_dim=custom_embedding_dimension
        )

        agent_chroma.add_memory("ChromaDB stores vectors for compact memory.")
        agent_chroma.add_memory("It can be persistent if a path is provided.")

        query_chroma_text = "How is ChromaDB used in this setup?"
        logging.info(f"Querying Chroma-backed agent: '{query_chroma_text}'")
        response_chroma = agent_chroma.query(query_chroma_text)
        logging.info(f"Response: {response_chroma}")

        agent_save_path_chroma = os.path.join(temp_demo_dir, "my_chroma_agent")
        logging.info(f"Saving Chroma-backed agent to: {agent_save_path_chroma}")
        agent_chroma.save_agent(agent_save_path_chroma)

        # For loading, Chroma store needs to be re-initialized with the same path
        # The vector_store_path saved in agent_config.json by agent.save_agent()
        # will point to where Chroma adapter should store its specific files,
        # which in this case is the path passed to its constructor.
        # The CLI's _create_dependencies_from_config handles this.
        # Here, we manually re-create it.

        # Path where Chroma data was actually persisted by the store within the agent's save structure
        # (agent_save_path_chroma / "vector_store" / "chroma_db_storage" (if it nests like that))
        # The Chroma adapter's `path` argument refers to the directory where Chroma itself will store its files.
        # If agent.save_agent() tells chroma_store.persist(agent_save_path + "/vector_store/chroma_specific"),
        # then that's the path to use.
        # Current Chroma adapter's persist is a no-op, relies on client's path.
        # So, we need to point to the original chroma_storage_path for the client.
        # The agent_config.json will store `vector_store_config: {"path": chroma_storage_path, ...}`

        # To correctly reload, we need the path that was initially used for Chroma.
        # This path is stored in agent_config.json by agent.save_agent().
        # Let's simulate reading that config to get the path.
        with open(os.path.join(agent_save_path_chroma, "agent_config.json"), "r") as f:
            saved_agent_conf = json.load(f)
        reloaded_chroma_path = saved_agent_conf.get("vector_store_config", {}).get("path")
        reloaded_collection_name = saved_agent_conf.get("vector_store_config", {}).get("collection_name")

        if reloaded_chroma_path and reloaded_collection_name:
            logging.info(f"Reloading Chroma store from path: {reloaded_chroma_path} and collection: {reloaded_collection_name}")
            reloaded_chroma_store = ChromaVectorStoreAdapter(
                collection_name=reloaded_collection_name,
                path=reloaded_chroma_path
            )
            loaded_agent_chroma = Agent.load_agent(
                agent_dir_path=agent_save_path_chroma,
                vector_store_instance=reloaded_chroma_store,
                embedding_fn=my_simple_embedding_fn
            )
            logging.info(f"Querying loaded Chroma agent: {loaded_agent_chroma.query(query_chroma_text)}")
        else:
            logging.warning("Could not determine Chroma path/collection from saved config for reload demo.")
        print("\n")
    else:
        print("ChromaDB not installed or adapter not available. Skipping ChromaVectorStoreAdapter demo.\n")


    if FAISS_INSTALLED and FaissVectorStoreAdapter:
        print("--- Demo: Agent with FaissVectorStoreAdapter and Custom Embeddings ---")
        # FAISS adapter saves its files into a directory specified in its `persist` method.
        # This directory is determined by agent.save_agent() typically as `agent_dir/vector_store/`.
        # For initialization, FaissVectorStoreAdapter doesn't need a path.

        faiss_store = FaissVectorStoreAdapter(embedding_dim=custom_embedding_dimension)

        agent_faiss = Agent(
            vector_store=faiss_store,
            embedding_fn=my_simple_embedding_fn,
            embedding_dim=custom_embedding_dimension
        )

        agent_faiss.add_memory("FAISS is efficient for similarity search.")
        agent_faiss.add_memory("It can handle large datasets with specific indexing.")

        query_faiss_text = "What is FAISS good for?"
        logging.info(f"Querying FAISS-backed agent: '{query_faiss_text}'")
        response_faiss = agent_faiss.query(query_faiss_text)
        logging.info(f"Response: {response_faiss}")

        agent_faiss_save_path = os.path.join(temp_demo_dir, "my_faiss_agent")
        logging.info(f"Saving FAISS-backed agent to: {agent_faiss_save_path}")
        agent_faiss.save_agent(agent_faiss_save_path) # This calls faiss_store.persist()

        # For loading FAISS, Agent.load_agent expects a FaissVectorStoreAdapter instance.
        # The CLI helper would normally create a new FaissVectorStoreAdapter instance,
        # and then Agent.load_agent calls `vector_store_instance.load(path_from_config)`.
        faiss_data_path_for_load = os.path.join(agent_faiss_save_path, "vector_store") # Path where agent saved it

        # Re-create an empty instance, then load data into it via Agent.load_agent
        reloaded_faiss_store_empty = FaissVectorStoreAdapter(embedding_dim=custom_embedding_dimension)

        loaded_faiss_agent = Agent.load_agent(
            agent_dir_path=agent_faiss_save_path,
            vector_store_instance=reloaded_faiss_store_empty, # This instance will be loaded by Agent.load_agent
            embedding_fn=my_simple_embedding_fn
        )
        logging.info(f"Querying loaded FAISS agent: {loaded_faiss_agent.query(query_faiss_text)}")
        print("\n")
    else:
        print("FAISS not installed or adapter not available. Skipping FaissVectorStoreAdapter demo.\n")

finally:
    try:
        shutil.rmtree(temp_demo_dir)
        logging.info(f"Cleaned up temporary demo directory: {temp_demo_dir}")
    except Exception as e:
        logging.error(f"Error cleaning up temporary demo directory {temp_demo_dir}: {e}")

print("Demo complete.")
import hashlib # Added for the custom embedding function
import sys # For sys.exit if core components missing
```
