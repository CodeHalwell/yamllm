import os
import dotenv
import pprint
from yamllm.core.llm import LLM

# Initialize pretty printer
pp = pprint.PrettyPrinter(indent=2, width=80)
dotenv.load_dotenv()

# Get the absolute path to the config file
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
config_path = os.path.join(root_dir, ".config_examples", "google_config.yaml")

llm = LLM(config_path=config_path, api_key=os.environ.get("GOOGLE_API_KEY"))

while True:
    try:          
        prompt = input("Human: ")
        if prompt.lower() == "exit":
            break
        response = llm.query(prompt)
        print("\nAI:")
        pp.pprint(response)
        print()
        
    except FileNotFoundError as e:
        pp.pprint(f"Configuration file not found: {e}")
    except ValueError as e:
        pp.pprint(f"Configuration error: {e}")
    except Exception as e:
        pp.pprint(f"An error occurred: {e}")
