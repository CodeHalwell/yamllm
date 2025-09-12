import os
import random
import dotenv
from yamllm import AnthropicAI
from yamllm.ui.chat import RichChatSession


def main() -> None:
    dotenv.load_dotenv()
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.join(root_dir, ".config_examples", "anthropic", "anthropic_config.yaml")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Missing ANTHROPIC_API_KEY.")
        return
    llm = AnthropicAI(config_path=config_path, api_key=api_key)
    style = random.choice(["bubble", "minimal", "compact"])  # randomized
    RichChatSession(llm, style=style).run()


if __name__ == "__main__":
    main()
