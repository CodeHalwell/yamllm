import os
import logging
import dotenv
from yamllm.src.yamllm.core.llm import LLM


dotenv.load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Get the absolute path to the config file
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.join(root_dir, ".config", "basic_config.yaml")

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

