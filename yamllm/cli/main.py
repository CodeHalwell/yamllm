"""
Main CLI entry point for YAMLLM.

This module assembles all CLI commands from the modular submodules.
"""

import argparse
import sys
import os
import asyncio
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from yamllm.core.parser import parse_yaml_config
from yamllm.core.setup_wizard import SetupWizard
from yamllm.core.error_messages import help_system

from .tools import setup_tools_commands
from .config import setup_config_commands
from .chat import setup_chat_commands
from .memory import setup_memory_commands
from .agent import setup_agent_commands
from .advanced import setup_advanced_commands
from .p1_features import setup_p1_commands

__version__ = "0.1.12"

console = Console()


def show_status(args: argparse.Namespace) -> int:
    """Show system status and health checks."""
    console.print("\n[bold cyan]YAMLLM System Status[/bold cyan]\n")
    
    # Check Python version
    import platform
    console.print(f"Python: [green]{platform.python_version()}[/green]")
    console.print(f"YAMLLM: [green]{__version__}[/green]")
    
    # Check dependencies
    console.print("\n[bold]Dependencies:[/bold]")
    deps = [
        ('anthropic', 'Anthropic'),
        ('openai', 'OpenAI'),
        ('google.generativeai', 'Google AI'),
        ('mistralai', 'Mistral AI'),
        ('faiss', 'FAISS'),
        ('rich', 'Rich'),
    ]
    
    for module, name in deps:
        try:
            __import__(module)
            console.print(f"  ✓ {name}: [green]installed[/green]")
        except ImportError:
            console.print(f"  ✗ {name}: [red]not installed[/red]")
    
    # Check environment variables
    console.print("\n[bold]API Keys:[/bold]")
    keys = [
        'OPENAI_API_KEY',
        'ANTHROPIC_API_KEY',
        'GOOGLE_API_KEY',
        'MISTRAL_API_KEY',
    ]
    
    for key in keys:
        if os.getenv(key):
            console.print(f"  ✓ {key}: [green]set[/green]")
        else:
            console.print(f"  ✗ {key}: [yellow]not set[/yellow]")
    
    return 0


def list_providers(args: argparse.Namespace) -> int:
    """List supported LLM providers."""
    console.print("\n[bold cyan]Supported LLM Providers[/bold cyan]\n")
    
    providers = [
        ("openai", "OpenAI", "GPT-4, GPT-3.5", "OPENAI_API_KEY"),
        ("anthropic", "Anthropic", "Claude 3", "ANTHROPIC_API_KEY"),
        ("google", "Google", "Gemini Pro", "GOOGLE_API_KEY"),
        ("mistral", "Mistral AI", "Mistral Large/Medium/Small", "MISTRAL_API_KEY"),
        ("azure_openai", "Azure OpenAI", "GPT-4, GPT-3.5", "AZURE_OPENAI_API_KEY"),
        ("openrouter", "OpenRouter", "Multiple models", "OPENROUTER_API_KEY"),
        ("deepseek", "DeepSeek", "DeepSeek models", "DEEPSEEK_API_KEY"),
    ]
    
    table = Table(box=box.ROUNDED)
    table.add_column("Provider", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Models", style="yellow")
    table.add_column("API Key Env", style="green")
    
    for provider_id, name, models, env_key in providers:
        table.add_row(provider_id, name, models, env_key)
    
    console.print(table)
    
    if args.check:
        console.print("\n[bold]Checking connectivity...[/bold]")
        console.print("[yellow]Connectivity check not yet implemented[/yellow]")
    
    return 0


def show_quick_start(args: argparse.Namespace) -> int:
    """Show quick start guide."""
    guide = """
[bold cyan]YAMLLM Quick Start Guide[/bold cyan]

[bold]1. Install YAMLLM:[/bold]
   pip install yamllm-core

[bold]2. Set up your API key:[/bold]
   export OPENAI_API_KEY="your-key-here"

[bold]3. Create a configuration:[/bold]
   yamllm config create --provider openai --preset casual -o config.yaml

[bold]4. Start chatting:[/bold]
   yamllm chat --config config.yaml

[bold]Common Commands:[/bold]
   yamllm init              # Interactive setup wizard
   yamllm tools list        # See available tools
   yamllm config presets    # List configuration presets
   yamllm providers         # List supported providers
   yamllm status            # Check system status

[dim]For more help: yamllm --help or yamllm <command> --help[/dim]
"""
    console.print(Panel(guide, border_style="cyan"))
    return 0


def show_getting_started(args: argparse.Namespace) -> int:
    """Show comprehensive getting started guide."""
    return help_system.show_getting_started_guide()


def run_setup_wizard(args: argparse.Namespace) -> int:
    """Run interactive setup wizard."""
    try:
        wizard = SetupWizard()
        wizard.run()
        return 0
    except KeyboardInterrupt:
        console.print("\n[yellow]Setup cancelled[/yellow]")
        return 0
    except Exception as e:
        console.print(f"[red]✗ Setup failed: {e}[/red]")
        return 1


def diagnose(args: argparse.Namespace) -> int:
    """Run diagnostic checks."""
    console.print("\n[bold cyan]YAMLLM Diagnose[/bold cyan]\n")
    problems = 0
    
    try:
        # Basic environment info
        import platform
        console.print(f"Python: [green]{platform.python_version()}[/green]")
        console.print(f"YAMLLM: [green]{__version__}[/green]")

        # Config, if provided
        cfg = None
        if getattr(args, 'config', None):
            try:
                cfg = parse_yaml_config(args.config)
                console.print(f"Config: [green]{args.config}[/green]")
            except Exception as e:
                console.print(f"Config: [red]invalid ({e})[/red]")
                problems += 1

        # Check dependencies
        console.print("\n[bold]Checking dependencies...[/bold]")
        critical_deps = ['anthropic', 'openai', 'pyyaml', 'rich']
        for dep in critical_deps:
            try:
                __import__(dep)
                console.print(f"  ✓ {dep}")
            except ImportError:
                console.print(f"  ✗ {dep} [red]missing[/red]")
                problems += 1

        if problems == 0:
            console.print("\n[green]✓ No problems detected[/green]")
        else:
            console.print(f"\n[yellow]Found {problems} problem(s)[/yellow]")

        return 0 if problems == 0 else 1

    except Exception as e:
        console.print(f"[red]Diagnostic failed: {e}[/red]")
        return 1


def mcp_list(args: argparse.Namespace) -> int:
    """List MCP tools."""
    try:
        from yamllm.core.llm import LLM
        cfg = args.config
        llm = LLM(config_path=cfg, api_key=os.getenv("OPENAI_API_KEY") or "")
        if not llm.mcp_client:
            console.print("[yellow]No MCP connectors configured in this config.[/yellow]")
            return 0
        tools_by_conn = asyncio.run(llm.mcp_client.discover_all_tools(force_refresh=True))
        if not tools_by_conn:
            console.print("[yellow]No tools discovered from MCP connectors.[/yellow]")
            return 0
        for name, tools in tools_by_conn.items():
            console.print(f"\n[bold cyan]{name}[/bold cyan] ({len(tools)} tools)")
            for t in tools[:20]:
                console.print(f" • [bold]{t.get('name')}[/bold]: {t.get('description','')}")
        return 0
    except Exception as e:
        console.print(f"[red]✗ MCP list failed: {e}[/red]")
        return 1


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="yamllm", 
        description="YAMLLM - YAML-based Language Model Configuration & Execution",
        epilog="Use 'yamllm <command> --help' for command-specific help."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=False, help="Available commands")

    # Init command - Interactive setup wizard
    init_cmd = sub.add_parser("init", help="Interactive setup wizard for new users")
    init_cmd.set_defaults(func=run_setup_wizard)
    
    # Status command
    status_cmd = sub.add_parser("status", help="Show system status and health checks")
    status_cmd.set_defaults(func=show_status)
    
    # Providers command
    providers_cmd = sub.add_parser("providers", help="List supported LLM providers")
    providers_cmd.add_argument("--check", action="store_true", help="Check provider connectivity")
    providers_cmd.set_defaults(func=list_providers)
    
    # Set up modular commands
    setup_tools_commands(sub)
    setup_config_commands(sub)
    setup_chat_commands(sub)
    setup_memory_commands(sub)
    setup_agent_commands(sub)
    setup_advanced_commands(sub)
    setup_p1_commands(sub)
    
    # Quick start command
    quickstart_cmd = sub.add_parser("quickstart", help="Show quick start guide")
    quickstart_cmd.set_defaults(func=show_quick_start)
    
    # Getting started guide
    guide_cmd = sub.add_parser("guide", help="Show comprehensive getting started guide")
    guide_cmd.set_defaults(func=show_getting_started)
    
    # MCP commands
    mcp_cmd = sub.add_parser("mcp", help="MCP utilities")
    mcp_sub = mcp_cmd.add_subparsers(dest="mcp_action", help="MCP actions")
    mcp_list_cmd = mcp_sub.add_parser("list", help="List tools from configured MCP connectors")
    mcp_list_cmd.add_argument("--config", required=True, help="Path to YAML config file")
    mcp_list_cmd.set_defaults(func=mcp_list)

    # Diagnose command
    diag_cmd = sub.add_parser("diagnose", help="Run system diagnostics")
    diag_cmd.add_argument("--config", help="Optional config file to check")
    diag_cmd.set_defaults(func=diagnose)

    # Parse arguments
    args = parser.parse_args(argv)
    
    # If no command specified, show help
    if not args.command:
        parser.print_help()
        return 0
    
    # Execute command
    if hasattr(args, 'func'):
        try:
            return args.func(args)
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted[/yellow]")
            return 130
        except Exception as e:
            console.print(f"[red]✗ Error: {e}[/red]")
            return 1
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
