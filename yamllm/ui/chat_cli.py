import argparse
import os
from dotenv import load_dotenv
from rich.prompt import Prompt
from rich.table import Table
from rich import box
import json
from datetime import datetime

from yamllm.core.llm import LLM
from yamllm.ui.chat import TerminalUI
from yamllm.ui.themes import theme_manager

def run_chat(args: argparse.Namespace) -> int:
    load_dotenv()
    api_key = None
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            print(f"Environment variable {args.api_key_env} is not set.")
            return 1
    else:
        # Best-effort common env vars
        for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "MISTRAL_API_KEY", "OPENROUTER_API_KEY"):
            if os.environ.get(k):
                api_key = os.environ.get(k)
                break
    if not api_key:
        print("No API key found. Set --api-key-env or a common provider env var.")
        return 1

    llm = LLM(config_path=args.config, api_key=api_key)
    ui = TerminalUI(style=args.style)
    message_history = []

    def _handle_command(command_input: str) -> bool:
        parts = command_input[1:].split()
        if not parts:
            return True
        
        cmd = parts[0].lower()
        cmd_args = parts[1:]

        if cmd in ["help", "h"]:
            ui.console.print("Available commands: /help, /clear, /theme, /history, /save, /exit")
        elif cmd in ["clear", "cls"]:
            ui.console.clear()
            ui.print_header(provider=llm.provider, model=llm.model)
        elif cmd == "theme":
            if not cmd_args:
                ui.console.print(f"Available themes: {', '.join(theme_manager.list_themes().keys())}")
                return True
            theme_name = cmd_args[0]
            try:
                ui.set_theme(theme_name)
                ui.console.clear()
                ui.print_header(provider=llm.provider, model=llm.model)
                ui.console.print(f"Theme set to [bold]{theme_name}[/bold]")
            except ValueError as e:
                ui.console.print(f"[red]{e}[/red]")
        elif cmd == "history":
            if not message_history:
                ui.console.print("[dim]No message history yet[/dim]")
                return True
            
            table = Table(title="Message History", box=box.MINIMAL)
            table.add_column("#", style="dim", width=4)
            table.add_column("Role", style="cyan")
            table.add_column("Message", style="white")

            for i, msg in enumerate(message_history):
                role = "User" if i % 2 == 0 else "Assistant"
                display_msg = msg[:80] + "..." if len(msg) > 80 else msg
                table.add_row(str(i + 1), role, display_msg)
            
            ui.console.print(table)

        elif cmd == "save":
            if not message_history:
                ui.console.print("[dim]No conversation to save[/dim]")
                return True
            
            filename = cmd_args[0] if cmd_args else f"yamllm_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            try:
                with open(filename, 'w') as f:
                    json.dump(message_history, f, indent=2)
                ui.console.print(f"[green]✓ Conversation saved to {filename}[/green]")
            except Exception as e:
                ui.console.print(f"[red]✗ Failed to save conversation: {e}[/red]")

        else:
            ui.console.print(f"[red]Unknown command: {cmd}[/red]")
        return True

    ui.print_header(provider=llm.provider, model=llm.model)
    ui.console.print("Type '/help' for commands, or 'exit' to quit.\n", style="dim")

    while True:
        try:
            user_input = Prompt.ask(f"[bold {ui.theme.colors.primary}]You[/bold {ui.theme.colors.primary}]")
            if user_input.strip().lower() == "exit":
                break
            
            if user_input.startswith('/'):
                _handle_command(user_input)
                continue

            message_history.append(user_input)
            ui.print_user_message(user_input)
            
            if llm.output_stream:
                response = ui.stream_assistant_response(llm, user_input)
                message_history.append(response)
            else:
                response = llm.query(user_input)
                if response:
                    ui.print_assistant_message(response)
                    message_history.append(response)

            ui.console.print()  # spacer
        except KeyboardInterrupt:
            ui.console.print("\n[dim]Exiting…[/dim]")
            break
        except Exception as e:
            ui.console.print(f"[red]Error:[/red] {e}")

    return 0