import os
import random
import dotenv
from yamllm import DeepSeek
from yamllm.ui.components import YAMLLMConsole, StreamingDisplay


def main() -> None:
    dotenv.load_dotenv()
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.join(root_dir, ".config_examples", "deepseek", "deepseek_config.yaml")
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Missing DEEPSEEK_API_KEY.")
        return
    llm = DeepSeek(config_path=config_path, api_key=api_key)
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
