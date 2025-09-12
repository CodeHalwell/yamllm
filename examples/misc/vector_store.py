from yamllm.memory import VectorStore
from yamllm.core import LLM
"""
Vector store inspection demo.

Tip: If you change embedding models and hit a dimension mismatch,
use the CLI to migrate:
    yamllm migrate-index --store-path memory/vector_store --purge
"""
import os
import dotenv


dotenv.load_dotenv()

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
config_path = os.path.join(root_dir, ".config_examples", "basic_config_openai.yaml")

similar = LLM(config_path=config_path, api_key=os.environ.get("OPENAI_API_KEY"))
vector_store = VectorStore(store_path='memory/vector_store')
vectors, metadata = vector_store.get_vec_and_text()
print(f"Number of vectors: {len(vectors)}")
print(f"Vector dimension: {vectors.shape[1] if len(vectors) > 0 else 0}")
print(f"Number of metadata entries: {len(metadata)}")

print(similar.find_similar_messages('what is my name', k = 1))
