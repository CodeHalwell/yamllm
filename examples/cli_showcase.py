#!/usr/bin/env python3
"""
YAMLLM CLI Showcase - Demonstrates the enhanced CLI capabilities.

This example shows how to use the new CLI features including:
- Interactive setup wizard
- Configuration management  
- Tool management
- Theme system
- Status monitoring

Run this example with: uv run python examples/cli_showcase.py
"""

import subprocess
import sys
from pathlib import Path
from rich.console import Console

# Force Rich colors for beautiful output
console = Console(force_terminal=True, color_system="256")

def run_command(cmd, description):
    """Run a CLI command and display results."""
    console.print(f"\n{'='*60}")
    console.print(f"[bold green]🚀 {description}[/bold green]")
    console.print(f"[cyan]Command: yamllm {cmd}[/cyan]")
    console.print('='*60)
    
    try:
        result = subprocess.run(
            f"uv run python -m yamllm.cli {cmd}".split(),
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.stdout:
            print(result.stdout)  # Keep stdout as-is since it has Rich formatting
        if result.stderr:
            console.print(f"[red]Error: {result.stderr}[/red]")
            
    except subprocess.TimeoutExpired:
        console.print("[red]Command timed out[/red]")
    except Exception as e:
        console.print(f"[red]Error running command: {e}[/red]")

def main():
    """Demonstrate CLI features."""
    console.print("[bold magenta]🎯 YAMLLM Enhanced CLI Showcase[/bold magenta]")
    console.print("[dim]This demo shows the new CLI capabilities[/dim]")
    
    # Check if we're in the right directory
    if not Path("yamllm").exists():
        console.print("[red]❌ Please run this from the yamllm project root directory[/red]")
        sys.exit(1)
    
    # 1. Show system status
    run_command("status", "System Status Check")
    
    # 2. List providers
    run_command("providers", "Available LLM Providers")
    
    # 3. Show tools
    run_command("tools list", "Available Tools")
    
    # 4. Show tool packs
    run_command("tools list --pack web", "Web Tool Pack")
    
    # 5. List themes
    run_command("theme list", "Available UI Themes")
    
    # 6. Show current theme
    run_command("theme current", "Current Theme")
    
    # 7. Preview a theme
    run_command("theme preview synthwave", "Preview Synthwave Theme")
    
    # 8. Show configuration presets
    run_command("config presets", "Configuration Presets")
    
    # 9. Show quick start guide
    run_command("quickstart", "Quick Start Guide")
    
    console.print(f"\n{'='*60}")
    console.print("[bold green]🎉 CLI Showcase Complete![/bold green]")
    console.print("\n[bold]Try these commands yourself:[/bold]")
    console.print("• [cyan]yamllm init[/cyan]          - Interactive setup wizard")
    console.print("• [cyan]yamllm status[/cyan]        - System health check")  
    console.print("• [cyan]yamllm tools manage[/cyan]  - Interactive tool management")
    console.print("• [cyan]yamllm theme set <theme>[/cyan] - Change UI theme")
    console.print("• [cyan]yamllm config create --provider openai --preset coding[/cyan]")
    console.print('='*60)

if __name__ == "__main__":
    main()