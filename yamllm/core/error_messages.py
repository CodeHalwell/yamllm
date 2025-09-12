"""
Enhanced error messages and help system for YAMLLM.

This module provides user-friendly error messages with actionable suggestions
and context-aware help.
"""

import os
from rich.console import Console
from rich.panel import Panel
from rich import box

console = Console()


class ErrorHelper:
    """Provides enhanced error messages and help suggestions."""
    
    @staticmethod
    def missing_api_key_error(provider: str) -> None:
        """Show helpful message for missing API key."""
        provider_info = {
            "openai": {
                "env_var": "OPENAI_API_KEY",
                "url": "https://platform.openai.com/api-keys",
                "description": "OpenAI API keys"
            },
            "anthropic": {
                "env_var": "ANTHROPIC_API_KEY", 
                "url": "https://console.anthropic.com/",
                "description": "Anthropic API keys"
            },
            "google": {
                "env_var": "GOOGLE_API_KEY",
                "url": "https://makersuite.google.com/app/apikey",
                "description": "Google AI Studio API keys"
            },
            "mistral": {
                "env_var": "MISTRAL_API_KEY",
                "url": "https://console.mistral.ai/",
                "description": "Mistral API keys"
            }
        }
        
        info = provider_info.get(provider.lower(), {
            "env_var": f"{provider.upper()}_API_KEY",
            "url": "provider's website",
            "description": f"{provider} API keys"
        })
        
        error_content = [
            f"[red]✗ Missing API key for {provider}[/red]\n",
            f"[bold]Environment variable needed:[/bold] {info['env_var']}",
            f"[bold]Get your API key:[/bold] {info['url']}",
            "",
            "[bold]To set your API key:[/bold]",
            f"• Export: [dim]export {info['env_var']}=your-key-here[/dim]",
            f"• Or add to .env file: [dim]{info['env_var']}=your-key-here[/dim]",
            "",
            "[dim]Note: Never commit API keys to version control[/dim]"
        ]
        
        panel = Panel(
            "\n".join(error_content),
            title="[red]API Key Required[/red]",
            border_style="red",
            box=box.ROUNDED
        )
        console.print(panel)
    
    @staticmethod
    def config_file_not_found_error(config_path: str) -> None:
        """Show helpful message for missing config file."""
        # Suggest nearby config files
        suggestions = []
        
        # Check for example configs
        if os.path.exists(".config_examples"):
            for root, dirs, files in os.walk(".config_examples"):
                for file in files:
                    if file.endswith(".yaml"):
                        suggestions.append(os.path.join(root, file))
        
        error_content = [
            f"[red]✗ Configuration file not found: {config_path}[/red]\n",
            "[bold]Solutions:[/bold]",
            "1. Create a new config:",
            "   [dim]yamllm config create --provider openai --preset casual[/dim]",
            "",
            "2. Use an example config:",
        ]
        
        if suggestions:
            for suggestion in suggestions[:3]:  # Show first 3
                error_content.append(f"   [dim]yamllm chat --config {suggestion}[/dim]")
        else:
            error_content.append("   [dim](No example configs found)[/dim]")
        
        error_content.extend([
            "",
            "3. Validate an existing config:",
            "   [dim]yamllm config validate path/to/config.yaml[/dim]"
        ])
        
        panel = Panel(
            "\n".join(error_content),
            title="[red]Configuration File Not Found[/red]",
            border_style="red",
            box=box.ROUNDED
        )
        console.print(panel)
    
    @staticmethod
    def invalid_config_error(error_msg: str, config_path: str) -> None:
        """Show helpful message for invalid config."""
        common_fixes = {
            "provider": [
                "Check the provider name is one of: openai, anthropic, google, mistral",
                "Ensure provider.name and provider.model are set"
            ],
            "model": [
                "Check the model name is valid for your provider",
                "Use 'yamllm config models' to see available models"
            ],
            "yaml": [
                "Check YAML syntax with proper indentation",
                "Ensure all strings are properly quoted",
                "Check for missing colons or commas"
            ],
            "tools": [
                "Check tool names are valid",
                "Use 'yamllm tools' to see available tools",
                "Ensure tool packs exist"
            ]
        }
        
        # Try to identify the issue type
        issue_type = None
        if "provider" in error_msg.lower():
            issue_type = "provider"
        elif "model" in error_msg.lower():
            issue_type = "model"
        elif "yaml" in error_msg.lower() or "syntax" in error_msg.lower():
            issue_type = "yaml"
        elif "tool" in error_msg.lower():
            issue_type = "tools"
        
        error_content = [
            f"[red]✗ Invalid configuration: {config_path}[/red]",
            f"[red]{error_msg}[/red]\n",
            "[bold]Common fixes:[/bold]"
        ]
        
        if issue_type and issue_type in common_fixes:
            for fix in common_fixes[issue_type]:
                error_content.append(f"• {fix}")
        else:
            # Generic fixes
            error_content.extend([
                "• Check YAML syntax and indentation",
                "• Ensure all required fields are present",
                "• Use 'yamllm config validate' for detailed validation"
            ])
        
        error_content.extend([
            "",
            "[bold]Helpful commands:[/bold]",
            f"• Validate: [dim]yamllm config validate {config_path}[/dim]",
            "• Create new: [dim]yamllm config create --provider <provider> --preset casual[/dim]",
            "• See examples: [dim]ls .config_examples/[/dim]"
        ])
        
        panel = Panel(
            "\n".join(error_content),
            title="[red]Configuration Error[/red]",
            border_style="red",
            box=box.ROUNDED
        )
        console.print(panel)
    
    @staticmethod
    def connection_error(provider: str, error_msg: str) -> None:
        """Show helpful message for connection errors."""
        troubleshooting = [
            "1. Check your internet connection",
            "2. Verify your API key is correct and active",
            "3. Check if the provider service is down",
            "4. Try again in a few minutes"
        ]
        
        if "timeout" in error_msg.lower():
            troubleshooting.extend([
                "5. Increase timeout in config (request.timeout)",
                "6. Check for network proxies or firewalls"
            ])
        elif "unauthorized" in error_msg.lower() or "401" in error_msg:
            troubleshooting = [
                "1. Check your API key is correct",
                "2. Verify your API key has the required permissions",
                "3. Check if your API key has expired",
                f"4. Generate a new API key from {provider} dashboard"
            ]
        elif "rate limit" in error_msg.lower() or "429" in error_msg:
            troubleshooting = [
                "1. Wait a few minutes and try again",
                "2. Reduce request frequency",
                "3. Check your plan's rate limits",
                "4. Consider upgrading your plan if needed"
            ]
        
        error_content = [
            f"[red]✗ Connection error with {provider}[/red]",
            f"[red]{error_msg}[/red]\n",
            "[bold]Troubleshooting steps:[/bold]"
        ]
        
        for step in troubleshooting:
            error_content.append(f"  {step}")
        
        error_content.extend([
            "",
            "[dim]If the problem persists, check the provider's status page[/dim]"
        ])
        
        panel = Panel(
            "\n".join(error_content),
            title="[red]Connection Error[/red]",
            border_style="red",
            box=box.ROUNDED
        )
        console.print(panel)
    
    @staticmethod
    def tool_error(tool_name: str, error_msg: str) -> None:
        """Show helpful message for tool errors."""
        common_solutions = {
            "weather": [
                "Set WEATHER_API_KEY environment variable",
                "Get free API key from OpenWeatherMap.org"
            ],
            "web_search": [
                "Check your internet connection",
                "DuckDuckGo search is free and doesn't need API key",
                "For enhanced search, set SERPAPI_API_KEY"
            ],
            "web_scraper": [
                "Check the URL is accessible",
                "Some sites block automated requests",
                "Check your internet connection"
            ],
            "calculator": [
                "Check your mathematical expression syntax",
                "Use standard operators: +, -, *, /, ^, sqrt(), log()"
            ]
        }
        
        error_content = [
            f"[red]✗ Tool '{tool_name}' failed[/red]",
            f"[red]{error_msg}[/red]\n"
        ]
        
        if tool_name in common_solutions:
            error_content.append("[bold]Possible solutions:[/bold]")
            for solution in common_solutions[tool_name]:
                error_content.append(f"• {solution}")
        else:
            error_content.extend([
                "[bold]General troubleshooting:[/bold]",
                "• Check if the tool requires API keys",
                "• Verify internet connection for network tools",
                "• Check input parameters are valid"
            ])
        
        error_content.extend([
            "",
            "[bold]Tool management:[/bold]",
            "• List tools: [dim]yamllm tools[/dim]",
            "• Disable in config: [dim]tools.enabled: false[/dim]"
        ])
        
        panel = Panel(
            "\n".join(error_content),
            title="[red]Tool Error[/red]",
            border_style="red",
            box=box.ROUNDED
        )
        console.print(panel)


class HelpSystem:
    """Enhanced help and guidance system."""
    
    @staticmethod
    def show_command_help(command: str) -> None:
        """Show detailed help for a specific command."""
        help_content = {
            "chat": {
                "description": "Start an interactive chat session with an LLM",
                "examples": [
                    "yamllm chat --config myconfig.yaml",
                    "yamllm chat --config .config_examples/openai/basic_config_openai.yaml",
                    "yamllm chat --config myconfig.yaml --style minimal"
                ],
                "tips": [
                    "Use Ctrl+C to interrupt a response",
                    "Type 'exit' to end the session",
                    "Chat history is saved if memory is enabled"
                ]
            },
            "config": {
                "description": "Manage configuration files",
                "examples": [
                    "yamllm config create --provider openai --preset casual",
                    "yamllm config validate myconfig.yaml",
                    "yamllm config presets",
                    "yamllm config models"
                ],
                "tips": [
                    "Start with presets for common use cases",
                    "Validate configs before using them",
                    "Keep sensitive info in environment variables"
                ]
            },
            "status": {
                "description": "Check system health and configuration",
                "examples": [
                    "yamllm status"
                ],
                "tips": [
                    "Run before troubleshooting issues",
                    "Shows which API keys are configured",
                    "Checks for required dependencies"
                ]
            }
        }
        
        if command in help_content:
            info = help_content[command]
            
            content = [
                f"[bold]{info['description']}[/bold]\n",
                "[bold]Examples:[/bold]"
            ]
            
            for example in info['examples']:
                content.append(f"  [dim]{example}[/dim]")
            
            content.extend([
                "",
                "[bold]Tips:[/bold]"
            ])
            
            for tip in info['tips']:
                content.append(f"• {tip}")
            
            panel = Panel(
                "\n".join(content),
                title=f"[cyan]Help: {command}[/cyan]",
                border_style="cyan",
                box=box.ROUNDED
            )
            console.print(panel)
    
    @staticmethod
    def show_getting_started() -> None:
        """Show comprehensive getting started guide."""
        steps = [
            {
                "title": "1. Install YAMLLM",
                "content": "pip install yamllm-core",
                "note": "Already done if you're seeing this!"
            },
            {
                "title": "2. Get API Key", 
                "content": "Get an API key from your preferred provider:\n• OpenAI: https://platform.openai.com/api-keys\n• Anthropic: https://console.anthropic.com/\n• Google: https://makersuite.google.com/app/apikey",
                "note": "Choose based on your needs and budget"
            },
            {
                "title": "3. Set Environment Variable",
                "content": "export OPENAI_API_KEY=your-key-here",
                "note": "Replace with your actual API key"
            },
            {
                "title": "4. Create Configuration",
                "content": "yamllm config create --provider openai --preset casual",
                "note": "This creates a ready-to-use config file"
            },
            {
                "title": "5. Start Chatting",
                "content": "yamllm chat --config openai_casual_config.yaml",
                "note": "Begin your conversation!"
            }
        ]
        
        console.print("\n[bold cyan]Getting Started with YAMLLM[/bold cyan]\n")
        
        for step in steps:
            panel_content = [
                f"[bold]Command:[/bold] [dim]{step['content']}[/dim]",
                f"[bold]Note:[/bold] {step['note']}"
            ]
            
            panel = Panel(
                "\n".join(panel_content),
                title=step['title'],
                border_style="green",
                box=box.MINIMAL
            )
            console.print(panel)
        
        console.print("\n[bold]Need help?[/bold]")
        console.print("• Check status: [dim]yamllm status[/dim]")
        console.print("• List providers: [dim]yamllm providers[/dim]")
        console.print("• View presets: [dim]yamllm config presets[/dim]")
        console.print("• Get help: [dim]yamllm --help[/dim]")


# Global instances for easy access
error_helper = ErrorHelper()
help_system = HelpSystem()