"""
Chat interface CLI commands for YAMLLM.

This module contains chat-related CLI commands.
"""

import argparse
import os
from rich.console import Console

from yamllm.ui.chat_cli import run_chat as run_chat_ui

console = Console()


def run_chat(args: argparse.Namespace) -> int:
    """Run an interactive chat session."""
    try:
        # Extract API key from environment
        api_key_env = args.api_key_env
        if api_key_env:
            api_key = os.getenv(api_key_env)
            if not api_key:
                console.print(f"[red]✗ Environment variable {api_key_env} not set[/red]")
                return 1
        else:
            # Try common env vars
            api_key = (
                os.getenv("OPENAI_API_KEY") or
                os.getenv("ANTHROPIC_API_KEY") or
                os.getenv("GOOGLE_API_KEY") or
                ""
            )
        
        if not api_key:
            console.print("[red]✗ No API key found in environment variables[/red]")
            console.print("[dim]Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY[/dim]")
            console.print("[dim]Or use --api-key-env to specify a custom environment variable[/dim]")
            return 1
        
        # Run the chat UI
        run_chat_ui(
            config_path=args.config,
            api_key=api_key,
            style=args.style,
            enhanced=args.enhanced
        )
        
        return 0
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Chat session interrupted[/yellow]")
        return 0
    except Exception as e:
        console.print(f"[red]✗ Chat session failed: {e}[/red]")
        return 1


def setup_chat_commands(subparsers):
    """Set up chat-related CLI commands."""
    # Chat command
    chat = subparsers.add_parser("chat", help="Run an interactive chat session")
    chat.add_argument("--config", required=True, help="Path to YAML config file")
    chat.add_argument("--api-key-env", default=None, help="Env var name holding provider API key (e.g., OPENAI_API_KEY)")
    chat.add_argument("--style", default="bubble", help="Chat UI style: bubble|minimal|compact")
    chat.add_argument("--enhanced", action="store_true", help="Use enhanced chat interface with commands and shortcuts")
    chat.set_defaults(func=run_chat)

    # Run alias for chat (manifesto-aligned quick start)
    run_cmd = subparsers.add_parser("run", help="Alias for 'chat'")
    run_cmd.add_argument("--config", required=True, help="Path to YAML config file")
    run_cmd.add_argument("--api-key-env", default=None, help="Env var name holding provider API key (e.g., OPENAI_API_KEY)")
    run_cmd.add_argument("--style", default="bubble", help="Chat UI style: bubble|minimal|compact")
    run_cmd.add_argument("--enhanced", action="store_true", help="Use enhanced chat interface with commands and shortcuts")
    run_cmd.set_defaults(func=run_chat)
