import os
import random
import dotenv
from yamllm import OpenAIGPT
from yamllm.ui.components import YAMLLMConsole, StreamingDisplay

"""
Azure OpenAI example using the reusable chat UI.
"""


def main() -> None:
    dotenv.load_dotenv()
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config = os.path.join(root, ".config_examples", "azure_openai", "azure_openai_config.yaml")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not api_key:
        print("Missing AZURE_OPENAI_API_KEY.")
        return
    llm = OpenAIGPT(config_path=config, api_key=api_key)
    ui = YAMLLMConsole(theme=random.choice(["default", "monokai", "dracula"]))
    stream = StreamingDisplay(ui)
    ui.print_banner()
    ui.console.print("Type 'exit' to quit.\n", style="dim")
    while True:
        try:
            prompt = ui.prompt_user("You")
            if not prompt or prompt.strip().lower() == "exit":
                break
            ui.print_message("user", prompt)
            stream.start("assistant")
            llm.set_stream_callback(stream.update)
            try:
                llm.query(prompt)
            finally:
                stream.stop()
                llm.clear_stream_callback()
        except KeyboardInterrupt:
            ui.console.print("\n[dim]Exiting…[/dim]")
            break


if __name__ == "__main__":
    main()
