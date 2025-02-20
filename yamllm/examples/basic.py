from yamllm.core.llm import OpenAIGPT, GoogleGemini, DeepSeek
import os
import dotenv

dotenv.load_dotenv()

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
config_path = os.path.join(root_dir, ".config_examples", "google_config.yaml")

# Initialize LLM with config
llm = GoogleGemini(config_path=config_path, api_key=os.environ.get("GOOGLE_API_KEY"))

# Make a query
response = llm.query("Give me some boiler plate pytorch code please")