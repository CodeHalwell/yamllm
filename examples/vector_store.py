from yamllm.memory import VectorStore
from yamllm.core import LLM
import os
import dotenv


dotenv.load_dotenv()

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
config_path = os.path.join(root_dir, ".config_examples", "basic_config_openai.yaml")

similar = LLM(config_path=config_path, api_key=os.environ.get("OPENAI_API_KEY"))
vector_store = VectorStore()
vectors, metadata = vector_store.get_vec_and_text()
print(f"Number of vectors: {len(vectors)}")
print(f"Vector dimension: {vectors.shape[1] if len(vectors) > 0 else 0}")
print(f"Number of metadata entries: {len(metadata)}")

print(similar.find_similar_messages('I like to do activities I am passionate about', k = 1))