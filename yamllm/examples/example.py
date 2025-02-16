import os
import dotenv
import pprint
from yamllm.core.llm import LLM

"""
This script initializes a language model (LLM) using a configuration file and an API key, 
then enters a loop where it takes user input, queries the LLM with the input, and prints the response.
Modules:
    os: Provides a way of using operating system dependent functionality.
    dotenv: Loads environment variables from a .env file.
    pprint: Provides a capability to pretty-print data structures.
    yamllm.core.llm: Contains the LLM class for interacting with the language model.
Functions:
    None
Usage:
    Run the script and enter prompts when prompted. Type 'exit' to terminate the loop.
Exceptions:
    FileNotFoundError: Raised when the configuration file is not found.
    ValueError: Raised when there is a configuration error.
    Exception: Catches all other exceptions and prints an error message.
"""

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
