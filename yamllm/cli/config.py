"""
Configuration management CLI commands for YAMLLM.

This module contains all configuration-related CLI commands.
"""

import argparse
import os
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from yamllm.core.parser import parse_yaml_config
from yamllm.core.config_validator import ConfigValidator
from yamllm.core.config_templates import template_manager

console = Console()


def validate_config(args: argparse.Namespace) -> int:
    """Validate a YAML configuration file."""
    config_path = args.config
    
    if not os.path.exists(config_path):
        console.print(f"[red]✗ Config file not found: {config_path}[/red]")
        return 1
    
    try:
        console.print(f"\n[bold cyan]Validating {config_path}...[/bold cyan]\n")
        
        # Parse config
        config = parse_yaml_config(config_path)
        
        # Convert to dict
        try:
            config_dict = config.model_dump()
        except Exception:
            config_dict = config.dict()
        
        # Validate
        errors = ConfigValidator.validate_config(config_dict)
        
        if errors:
            console.print("[red]✗ Configuration has errors:[/red]\n")
            for error in errors:
                console.print(f"  • {error}")
            return 1
        
        console.print("[green]✓ Configuration is valid[/green]")
        
        # Show summary
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  Provider: {config.provider.name}")
        console.print(f"  Model: {config.provider.model}")
        console.print(f"  Temperature: {config.model_settings.temperature}")
        console.print(f"  Max Tokens: {config.model_settings.max_tokens}")
        console.print(f"  Streaming: {config.output.stream}")
        console.print(f"  Tools Enabled: {config.tools.enabled}")
        console.print(f"  Memory Enabled: {config.context.memory.enabled}")
        
        return 0
        
    except Exception as e:
        console.print(f"[red]✗ Validation failed: {e}[/red]")
        return 1


def create_config(args: argparse.Namespace) -> int:
    """Create a new configuration file."""
    provider = args.provider
    preset = args.preset
    output_path = args.output or f"{provider}_{preset}.yaml"
    
    console.print(f"\n[bold cyan]Creating configuration...[/bold cyan]\n")
    console.print(f"Provider: {provider}")
    console.print(f"Preset: {preset}")
    console.print(f"Output: {output_path}")
    
    try:
        # Get template
        template = template_manager.get_template(provider, preset)
        
        if not template:
            console.print(f"[red]✗ Template not found for {provider}/{preset}[/red]")
            return 1
        
        # Customize with args
        if args.model:
            template['provider']['model'] = args.model
        if args.temperature:
            template['model_settings']['temperature'] = args.temperature
        if args.max_tokens:
            template['model_settings']['max_tokens'] = args.max_tokens
        
        template['output']['stream'] = args.streaming
        
        if args.no_tools:
            template['tools']['enabled'] = False
        if args.no_memory:
            template['context']['memory']['enabled'] = False
        
        # Write to file
        import yaml
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(template, f, default_flow_style=False, sort_keys=False)
        
        console.print(f"\n[green]✓ Configuration created: {output_path}[/green]")
        console.print(f"\n[dim]Edit this file to customize your settings.[/dim]")
        console.print(f"[dim]Remember to set your API key in environment variables.[/dim]")
        
        return 0
        
    except Exception as e:
        console.print(f"[red]✗ Failed to create configuration: {e}[/red]")
        return 1


def list_presets(args: argparse.Namespace) -> int:
    """List available configuration presets."""
    console.print("\n[bold cyan]Available Configuration Presets[/bold cyan]\n")
    
    presets = template_manager.list_presets()
    
    table = Table(box=box.ROUNDED)
    table.add_column("Preset", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Use Case", style="yellow")
    
    for preset_name, preset_info in presets.items():
        table.add_row(
            preset_name,
            preset_info.get('description', 'No description'),
            preset_info.get('use_case', 'General')
        )
    
    console.print(table)
    console.print(f"\n[dim]Use 'yamllm config create --preset <preset>' to create a config[/dim]")
    
    return 0


def list_models(args: argparse.Namespace) -> int:
    """List available models by provider."""
    console.print("\n[bold cyan]Available Models[/bold cyan]\n")
    
    models_by_provider = {
        "openai": [
            ("gpt-4", "Most capable GPT-4 model"),
            ("gpt-4-turbo", "GPT-4 Turbo with vision"),
            ("gpt-3.5-turbo", "Fast and efficient"),
        ],
        "anthropic": [
            ("claude-3-opus-20240229", "Most capable Claude model"),
            ("claude-3-sonnet-20240229", "Balanced performance"),
            ("claude-3-haiku-20240307", "Fast and compact"),
        ],
        "google": [
            ("gemini-pro", "Google's flagship model"),
            ("gemini-pro-vision", "With vision capabilities"),
        ]
    }
    
    for provider, models in models_by_provider.items():
        console.print(f"\n[bold]{provider.upper()}[/bold]")
        
        table = Table(box=box.ROUNDED, show_header=False)
        table.add_column("Model", style="cyan")
        table.add_column("Description", style="white")
        
        for model_name, description in models:
            table.add_row(model_name, description)
        
        console.print(table)
    
    console.print(f"\n[dim]Use 'yamllm config create --model <model>' to specify a model[/dim]")
    
    return 0


def setup_config_commands(subparsers):
    """Set up config-related CLI commands."""
    config_cmd = subparsers.add_parser("config", help="Configuration management")
    config_sub = config_cmd.add_subparsers(dest="config_action", help="Config actions")
    
    # Config create
    create_cmd = config_sub.add_parser("create", help="Create a new configuration file")
    create_cmd.add_argument("--provider", required=True, choices=["openai", "anthropic", "google"],
                           help="LLM provider")
    create_cmd.add_argument("--preset", default="casual", choices=["casual", "coding", "research", "minimal"],
                           help="Configuration preset")
    create_cmd.add_argument("--model", help="Specific model to use")
    create_cmd.add_argument("--output", "-o", help="Output file path")
    create_cmd.add_argument("--temperature", type=float, default=0.7, help="Model temperature")
    create_cmd.add_argument("--max-tokens", type=int, default=1000, help="Maximum tokens")
    create_cmd.add_argument("--streaming", action="store_true", default=True, help="Enable streaming")
    create_cmd.add_argument("--no-streaming", action="store_false", dest="streaming", help="Disable streaming")
    create_cmd.add_argument("--no-tools", action="store_true", help="Disable tools")
    create_cmd.add_argument("--no-memory", action="store_true", help="Disable memory")
    create_cmd.set_defaults(func=create_config)
    
    # Config validate
    validate_cmd = config_sub.add_parser("validate", help="Validate a configuration file")
    validate_cmd.add_argument("config", help="Path to YAML config file")
    validate_cmd.set_defaults(func=validate_config)
    
    # Config presets
    presets_cmd = config_sub.add_parser("presets", help="List available configuration presets")
    presets_cmd.set_defaults(func=list_presets)
    
    # Config models
    models_cmd = config_sub.add_parser("models", help="List available models")
    models_cmd.set_defaults(func=list_models)
