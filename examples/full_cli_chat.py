"""Interactive CLI demo using yamllm's full UI, tools, thinking, and MCP stack.

Run with:
    uv run examples/full_cli_chat.py

Requires the example configuration at `.config_examples/full_stack_cli.yaml`
and at least one provider API key exported (OPENAI_API_KEY recommended). Optional
tool and MCP tokens can be supplied via the environment.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Optional
import re

from dotenv import load_dotenv

from yamllm.core.llm import LLM
from yamllm.ui.chat import TerminalUI


DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / ".config_examples/full_stack_cli.yaml"
DEFAULT_THEME = "default"
DEFAULT_STYLE = "bubble"


def _resolve_api_key(preferred_env: Optional[str]) -> Optional[str]:
    """Find an API key from the environment following repo conventions."""
    priority: Iterable[str]
    if preferred_env:
        priority = (preferred_env,)
    else:
        priority = (
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "MISTRAL_API_KEY",
            "DEEPSEEK_API_KEY",
            "OPENROUTER_API_KEY",
        )
    for env_var in priority:
        value = os.environ.get(env_var)
        if value:
            return value
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a fully loaded yamllm chat session")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to the yamllm YAML configuration (defaults to full_stack_cli.yaml)",
    )
    parser.add_argument(
        "--theme",
        default=DEFAULT_THEME,
        help="Terminal theme to apply (see yamllm/ui/themes)",
    )
    parser.add_argument(
        "--style",
        default=DEFAULT_STYLE,
        choices=["bubble", "compact"],
        help="Chat bubble style to use in the terminal UI",
    )
    parser.add_argument(
        "--api-key-env",
        help="Explicit environment variable to read for the provider API key",
    )
    args = parser.parse_args()

    load_dotenv()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}", file=sys.stderr)
        return 1

    api_key = _resolve_api_key(args.api_key_env)
    if not api_key:
        print(
            "Unable to locate a provider API key. Export one of OPENAI_API_KEY, "
            "ANTHROPIC_API_KEY, GOOGLE_API_KEY, MISTRAL_API_KEY, DEEPSEEK_API_KEY, or OPENROUTER_API_KEY.",
            file=sys.stderr,
        )
        return 1

    # Best-effort reminders for optional integrations
    optional_keys = {
        "WEATHER_API_KEY": "weather tool (OpenWeatherMap)",
        "SERPAPI_API_KEY": "SerpAPI web_search provider",
        "TAVILY_API_KEY": "Tavily search provider",
        "BING_SEARCH_API_KEY": "Bing web_search provider",
        "NOTES_MCP_TOKEN": "notes-http MCP connector",
    }
    missing_optional = [name for name, desc in optional_keys.items() if desc and not os.environ.get(name)]
    if missing_optional:
        ui_hint = ", ".join(missing_optional)
        print(
            "Tip: set optional environment variables for richer tooling ("
            f"{ui_hint}).",
            file=sys.stderr,
        )

    llm = LLM(config_path=str(config_path), api_key=api_key)

    ui = TerminalUI(style=args.style)
    try:
        ui.set_theme(args.theme)
    except ValueError as exc:
        print(f"Warning: {exc}. Falling back to default theme.")
        ui.set_theme(DEFAULT_THEME)

    ui.console.clear()
    ui.print_header(provider=llm.provider, model=llm.model)
    ui.console.print(
        "Streaming is enabled. Thinking traces, tool calls, and MCP interactions "
        "will render inline as they occur. Type '/help' for helper commands or 'exit' to quit.",
        style="dim",
    )
    if not any(conn.enabled for conn in getattr(llm.config.tools, "mcp_connectors", [])):
        ui.console.print(
            "[dim]MCP connectors are disabled by default. Enable them in the config after installing"
            " optional dependencies such as httpx[h2] and starting an MCP server.[/dim]",
        )

    message_history: list[str] = []

    def handle_command(command_input: str) -> bool:
        parts = command_input[1:].split()
        if not parts:
            return True
        name = parts[0].lower()
        if name in {"help", "h"}:
            ui.console.print(
                "Commands: /help, /clear, /theme <name>, /history, /save [file], /exit",
                style="dim",
            )
        elif name == "clear":
            ui.console.clear()
            ui.print_header(provider=llm.provider, model=llm.model)
        elif name == "theme" and len(parts) > 1:
            try:
                ui.set_theme(parts[1])
                ui.console.clear()
                ui.print_header(provider=llm.provider, model=llm.model)
                ui.console.print(f"Theme switched to {parts[1]}")
            except ValueError as exc:
                ui.console.print(f"[red]{exc}[/red]")
        elif name == "history":
            if not message_history:
                ui.console.print("[dim]No history yet[/dim]")
            else:
                for idx, msg in enumerate(message_history, 1):
                    role = "You" if idx % 2 == 1 else "Assistant"
                    ui.console.print(f"[bold]{role}[/bold]: {msg}")
        elif name == "save":
            if not message_history:
                ui.console.print("[dim]Nothing to save yet[/dim]")
            else:
                target = parts[1] if len(parts) > 1 else "yamllm_chat_transcript.txt"
                try:
                    Path(target).write_text("\n".join(message_history), encoding="utf-8")
                    ui.console.print(f"[green]Saved transcript to {target}[/green]")
                except OSError as exc:
                    ui.console.print(f"[red]Failed to save transcript: {exc}[/red]")
        elif name == "exit":
            return False
        else:
            ui.console.print(f"[red]Unknown command: {name}[/red]")
        return True

    while True:
        try:
            user_input = ui.console.input(f"[bold {ui.theme.colors.primary}]You[/bold {ui.theme.colors.primary}] ")
            stripped = user_input.strip()
            if not stripped:
                continue
            if stripped.lower() == "exit":
                break
            if stripped.startswith("/"):
                if not handle_command(stripped):
                    break
                continue

            message_history.append(stripped)
            ui.print_user_message(stripped)

            if llm.output_stream:
                reply = ui.stream_assistant_response(llm, stripped)
            else:
                reply = llm.query(stripped)
                if reply:
                    ui.print_assistant_message(reply)

            if reply:
                message_history.append(reply)
            ui.console.print()
        except KeyboardInterrupt:
            ui.console.print("\n[dim]Interrupted. Type 'exit' to quit next time.[/dim]")
            break
        except Exception as exc:
            ui.console.print(f"[red]Error:[/red] {exc}")

    llm.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
