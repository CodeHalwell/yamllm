import os
from llm import LLM  # Update the import path
import logging
import dotenv

dotenv.load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Get the absolute path to the config file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "..", "..", "examples", "basic_config.yaml")

    llm = LLM(config_path=config_path)

    llm.api_key = os.environ.get("OPENAI_API_KEY")


    llm.print_settings()
    
    while True:
        try:          
            prompt = input("Human: ")
            if prompt.lower() == "exit":
                break
            logger.info(f"\nSending basic query: {prompt}\n")
            response = llm.query(prompt)
            logger.info(f"\nResponse: {response}\n")

            
        except FileNotFoundError as e:
            logger.error(f"Configuration file not found: {e}")
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
        except Exception as e:
            logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()

