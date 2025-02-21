import os
import dotenv
from rich.console import Console
from yamllm.core.llm import GoogleGemini, MistralAI

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
console = Console()
dotenv.load_dotenv()

# Get the absolute path to the config file
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
config_path = os.path.join(root_dir, ".config_examples", "mistral_config.yaml")

llm = MistralAI(config_path=config_path, api_key=os.environ.get("MISTRAL_API_KEY"))

while True:
    try:          
        prompt = input("\nHuman: ")
        if prompt.lower() == "exit":
            break
        
        response = llm.query(prompt)
        if response is None:
            continue
        
    except FileNotFoundError as e:
        console.print(f"[red]Configuration file not found:[/red] {e}")
    except ValueError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
    except Exception as e:
        console.print(f"[red]An error occurred:[/red] {str(e)}")
