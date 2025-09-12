#!/usr/bin/env python3
"""
OpenAI Chat — Beautiful streaming UI in ~10 lines.
"""
import os
import random
from dotenv import load_dotenv
from yamllm import OpenAIGPT
from yamllm.ui.chat import RichChatSession


def main() -> None:
    load_dotenv()
    config = ".config_examples/openai/basic_config_openai.yaml"
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Missing OPENAI_API_KEY. Set it in your environment.")
        return
    llm = OpenAIGPT(config_path=config, api_key=api_key)
    style = random.choice(["bubble", "minimal", "compact"])  # randomized
    RichChatSession(llm, style=style).run()


if __name__ == "__main__":
    main()
