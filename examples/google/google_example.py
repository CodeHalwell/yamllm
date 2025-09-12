import os
import random
import dotenv
from yamllm import GoogleGemini
from yamllm.ui.chat import RichChatSession


def main() -> None:
    dotenv.load_dotenv()
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.join(root_dir, ".config_examples", "google", "google_config.yaml")
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Missing GOOGLE_API_KEY.")
        return
    llm = GoogleGemini(config_path=config_path, api_key=api_key)
    style = random.choice(["bubble", "minimal", "compact"])  # randomized
    RichChatSession(llm, style=style).run()


if __name__ == "__main__":
    main()
