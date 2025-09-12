"""
Interactive setup wizard for YAMLLM.

This module provides a step-by-step guided setup experience for new users,
helping them configure their environment and create their first configuration.
"""

import os

from rich.console import Console
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

from yamllm.core.config_templates import template_manager

console = Console()


class SetupWizard:
    """Interactive setup wizard for YAMLLM."""
    
    def __init__(self):
        self.config_data = {}
        self.selected_provider = None
        self.selected_preset = None
        self.api_key_status = {}
        
    def run(self) -> bool:
        """Run the complete setup wizard."""
        console.clear()
        
        # Welcome
        self._show_welcome()
        
        # Check if user wants to proceed
        if not Confirm.ask("\n[bold]Ready to get started?[/bold]", default=True):
            console.print("[dim]Setup cancelled. Run 'yamllm init' anytime to start again.[/dim]")
            return False
        
        try:
            # Step 1: Provider selection
            if not self._step_provider_selection():
                return False
            
            # Step 2: API key setup
            if not self._step_api_key_setup():
                return False
            
            # Step 3: Preset selection
            if not self._step_preset_selection():
                return False
            
            # Step 4: Advanced configuration (optional)
            self._step_advanced_config()
            
            # Step 5: Create configuration file
            if not self._step_create_config():
                return False
            
            # Step 6: Test setup
            self._step_test_setup()
            
            # Success
            self._show_success()
            return True
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Setup interrupted. Run 'yamllm init' to start again.[/yellow]")
            return False
        except Exception as e:
            console.print(f"\n[red]Setup failed: {e}[/red]")
            return False
    
    def _show_welcome(self):
        """Show welcome message and overview."""
        banner = """
[bold cyan]
╔══════════════════════════════════════════════════════════════════════════════╗
║                         YAMLLM Setup Wizard                                 ║
║                                                                              ║
║           Welcome! Let's get you set up with YAMLLM in just a few steps.    ║
╚══════════════════════════════════════════════════════════════════════════════╝
[/bold cyan]
"""
        console.print(banner)
        
        steps = [
            "🔧 Choose your AI provider",
            "🔑 Set up API credentials", 
            "⚙️  Select configuration preset",
            "🎨 Customize settings (optional)",
            "📁 Create configuration file",
            "✅ Test your setup"
        ]
        
        console.print("[bold]What we'll do:[/bold]")
        for step in steps:
            console.print(f"  {step}")
        
        console.print("\n[dim]This will take about 2-3 minutes. You can cancel anytime with Ctrl+C.[/dim]")
    
    def _step_provider_selection(self) -> bool:
        """Step 1: Provider selection with detailed information."""
        console.print("\n" + "="*50)
        console.print("[bold cyan]Step 1: Choose Your AI Provider[/bold cyan]")
        console.print("="*50)
        
        # Show provider information
        providers_info = {
            "1": {
                "name": "openai",
                "display": "OpenAI (GPT-4, GPT-3.5)",
                "description": "Industry leader with high-quality models",
                "pros": ["Excellent quality", "Wide model selection", "Good documentation"],
                "cons": ["Higher cost", "Rate limits"],
                "cost": "$$$ (Premium)"
            },
            "2": {
                "name": "anthropic", 
                "display": "Anthropic (Claude)",
                "description": "Safety-focused AI with strong reasoning",
                "pros": ["Large context window", "Safety-focused", "Great for analysis"],
                "cons": ["Limited availability", "Newer provider"],
                "cost": "$$ (Moderate)"
            },
            "3": {
                "name": "google",
                "display": "Google (Gemini)", 
                "description": "Google's multimodal AI with free tier",
                "pros": ["Free tier available", "Multimodal", "Fast"],
                "cons": ["Newer models", "Limited features"],
                "cost": "$ (Free tier + paid)"
            }
        }
        
        # Display provider table
        table = Table(title="Available Providers", box=box.ROUNDED)
        table.add_column("Option", style="cyan", width=6)
        table.add_column("Provider", style="bold", width=20)
        table.add_column("Description", width=30)
        table.add_column("Cost", style="yellow", width=15)
        
        for option, info in providers_info.items():
            table.add_row(option, info["display"], info["description"], info["cost"])
        
        console.print(table)
        
        # Get selection
        while True:
            choice = Prompt.ask("\n[bold]Choose a provider[/bold]", choices=["1", "2", "3"])
            
            selected_info = providers_info[choice]
            
            # Show detailed info
            console.print(f"\n[bold]You selected: {selected_info['display']}[/bold]")
            console.print(f"[dim]{selected_info['description']}[/dim]")
            
            console.print("\n[green]✓ Pros:[/green]")
            for pro in selected_info["pros"]:
                console.print(f"  • {pro}")
                
            console.print("\n[yellow]⚠ Considerations:[/yellow]")
            for con in selected_info["cons"]:
                console.print(f"  • {con}")
            
            if Confirm.ask(f"\n[bold]Use {selected_info['display']}?[/bold]", default=True):
                self.selected_provider = selected_info["name"]
                console.print(f"[green]✓ Provider selected: {selected_info['display']}[/green]")
                return True
    
    def _step_api_key_setup(self) -> bool:
        """Step 2: API key setup with validation."""
        console.print("\n" + "="*50)
        console.print("[bold cyan]Step 2: Set Up API Credentials[/bold cyan]")
        console.print("="*50)
        
        provider_info = {
            "openai": {
                "env_var": "OPENAI_API_KEY",
                "url": "https://platform.openai.com/api-keys",
                "instructions": [
                    "1. Go to https://platform.openai.com/api-keys",
                    "2. Click 'Create new secret key'",
                    "3. Copy the key (starts with 'sk-')",
                    "4. Paste it below"
                ]
            },
            "anthropic": {
                "env_var": "ANTHROPIC_API_KEY",
                "url": "https://console.anthropic.com/",
                "instructions": [
                    "1. Go to https://console.anthropic.com/",
                    "2. Navigate to 'API Keys'",
                    "3. Create a new key",
                    "4. Copy and paste it below"
                ]
            },
            "google": {
                "env_var": "GOOGLE_API_KEY",
                "url": "https://makersuite.google.com/app/apikey",
                "instructions": [
                    "1. Go to https://makersuite.google.com/app/apikey",
                    "2. Click 'Create API Key'",
                    "3. Copy the generated key",
                    "4. Paste it below"
                ]
            }
        }
        
        info = provider_info[self.selected_provider]
        env_var = info["env_var"]
        
        # Check if API key already exists
        existing_key = os.getenv(env_var)
        if existing_key:
            console.print(f"[green]✓ API key already set in {env_var}[/green]")
            if Confirm.ask("Use existing API key?", default=True):
                return True
        
        # Show instructions
        panel_content = []
        panel_content.append("[bold]Get your API key:[/bold]")
        for instruction in info["instructions"]:
            panel_content.append(instruction)
        
        panel_content.extend([
            "",
            f"[dim]URL: {info['url']}[/dim]"
        ])
        
        panel = Panel(
            "\n".join(panel_content),
            title="API Key Instructions",
            border_style="cyan",
            box=box.ROUNDED
        )
        console.print(panel)
        
        # Get API key
        while True:
            api_key = Prompt.ask(
                f"\n[bold]Enter your {self.selected_provider.title()} API key[/bold]",
                password=True
            )
            
            if not api_key.strip():
                console.print("[red]API key cannot be empty[/red]")
                continue
            
            # Basic validation
            valid = True
            if self.selected_provider == "openai" and not api_key.startswith("sk-"):
                console.print("[yellow]⚠ OpenAI keys usually start with 'sk-'[/yellow]")
                valid = Confirm.ask("Continue anyway?", default=False)
            
            if valid:
                # Set environment variable for current session
                os.environ[env_var] = api_key
                console.print("[green]✓ API key set for this session[/green]")
                
                # Provide instructions for permanent setup
                console.print("\n[bold]To make this permanent, add to your shell profile:[/bold]")
                console.print(f"[dim]export {env_var}={api_key}[/dim]")
                console.print("\n[bold]Or add to a .env file:[/bold]")
                console.print(f"[dim]{env_var}={api_key}[/dim]")
                
                return True
    
    def _step_preset_selection(self) -> bool:
        """Step 3: Preset selection with detailed descriptions."""
        console.print("\n" + "="*50)
        console.print("[bold cyan]Step 3: Choose Configuration Preset[/bold cyan]")
        console.print("="*50)
        
        presets_info = {
            "1": {
                "name": "casual",
                "display": "Casual Chat",
                "description": "Perfect for everyday conversations",
                "features": ["Basic tools (calculator, weather, web search)", "Conversational tone", "Memory enabled", "Balanced settings"],
                "good_for": "General questions, casual conversations, learning"
            },
            "2": {
                "name": "coding",
                "display": "Programming Assistant", 
                "description": "Optimized for software development",
                "features": ["Development tools (files, text processing)", "Code-focused prompts", "Extended context", "Higher token limits"],
                "good_for": "Coding help, debugging, architecture discussions"
            },
            "3": {
                "name": "research",
                "display": "Research Assistant",
                "description": "Built for analysis and research tasks", 
                "features": ["Research tools (web scraping, analysis)", "Citation-focused", "Extended memory", "Unrestricted web access"],
                "good_for": "Research, analysis, fact-checking, writing"
            },
            "4": {
                "name": "minimal",
                "display": "Minimal Setup",
                "description": "Lightweight configuration with basics only",
                "features": ["No tools", "Basic prompts", "No memory", "Lower resource usage"],
                "good_for": "Simple Q&A, low resource usage, testing"
            }
        }
        
        # Display preset options
        for option, preset in presets_info.items():
            console.print(f"\n[bold cyan]{option}. {preset['display']}[/bold cyan]")
            console.print(f"[dim]{preset['description']}[/dim]")
            console.print(f"[bold]Good for:[/bold] {preset['good_for']}")
            
            console.print("[bold]Features:[/bold]")
            for feature in preset['features']:
                console.print(f"  • {feature}")
        
        # Get selection
        while True:
            choice = Prompt.ask("\n[bold]Choose a preset[/bold]", choices=["1", "2", "3", "4"])
            
            selected_preset = presets_info[choice]
            console.print(f"\n[bold]You selected: {selected_preset['display']}[/bold]")
            console.print(f"[dim]{selected_preset['description']}[/dim]")
            
            if Confirm.ask("Use this preset?", default=True):
                self.selected_preset = selected_preset["name"]
                console.print(f"[green]✓ Preset selected: {selected_preset['display']}[/green]")
                return True
    
    def _step_advanced_config(self):
        """Step 4: Optional advanced configuration."""
        console.print("\n" + "="*50)
        console.print("[bold cyan]Step 4: Advanced Settings (Optional)[/bold cyan]")
        console.print("="*50)
        
        if not Confirm.ask("Customize advanced settings?", default=False):
            console.print("[dim]Using preset defaults[/dim]")
            return
        
        console.print("\n[bold]Advanced Configuration:[/bold]")
        
        # Temperature
        default_temp = 0.7
        temp = FloatPrompt.ask(
            f"Model temperature (0.0-2.0, default {default_temp})",
            default=default_temp
        )
        self.config_data["temperature"] = temp
        
        # Max tokens
        default_tokens = 1000 if self.selected_preset != "coding" else 2000
        tokens = IntPrompt.ask(
            f"Maximum tokens (default {default_tokens})",
            default=default_tokens
        )
        self.config_data["max_tokens"] = tokens
        
        # Streaming
        streaming = Confirm.ask("Enable streaming responses?", default=True)
        self.config_data["streaming"] = streaming
        
        # Tools
        if self.selected_preset != "minimal":
            enable_tools = Confirm.ask("Enable tools?", default=True)
            self.config_data["tools_enabled"] = enable_tools
        
        # Memory
        enable_memory = Confirm.ask("Enable conversation memory?", default=True)
        self.config_data["memory_enabled"] = enable_memory
        
        console.print("[green]✓ Advanced settings configured[/green]")
    
    def _step_create_config(self) -> bool:
        """Step 5: Create configuration file."""
        console.print("\n" + "="*50)
        console.print("[bold cyan]Step 5: Create Configuration File[/bold cyan]")
        console.print("="*50)
        
        # Suggest filename
        default_filename = f"{self.selected_provider}_{self.selected_preset}_config.yaml"
        
        filename = Prompt.ask(
            "Configuration filename",
            default=default_filename
        )
        
        # Check if file exists
        if os.path.exists(filename):
            if not Confirm.ask(f"File {filename} exists. Overwrite?", default=False):
                filename = Prompt.ask("Enter a different filename")
        
        try:
            # Create configuration
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Creating configuration...", total=100)
                
                config = template_manager.create_config(
                    provider=self.selected_provider,
                    preset=self.selected_preset,
                    output_path=filename,
                    **self.config_data
                )
                progress.update(task, completed=100)
            
            console.print(f"[green]✓ Configuration created: {filename}[/green]")
            self.config_filename = filename
            return True
            
        except Exception as e:
            console.print(f"[red]✗ Failed to create configuration: {e}[/red]")
            return False
    
    def _step_test_setup(self):
        """Step 6: Test the setup."""
        console.print("\n" + "="*50)
        console.print("[bold cyan]Step 6: Test Your Setup[/bold cyan]")
        console.print("="*50)
        
        if not Confirm.ask("Test the configuration?", default=True):
            return
        
        try:
            # Test config validation
            from yamllm.core.parser import parse_yaml_config
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Testing configuration...", total=100)
                
                config = parse_yaml_config(self.config_filename)
                progress.update(task, completed=50)
                
                # Basic validation
                assert config.provider.name == self.selected_provider
                progress.update(task, completed=100)
            
            console.print("[green]✓ Configuration test passed[/green]")
            
        except Exception as e:
            console.print(f"[red]✗ Configuration test failed: {e}[/red]")
            console.print("[dim]You can still use the configuration, but there might be issues.[/dim]")
    
    def _show_success(self):
        """Show success message and next steps."""
        console.print("\n" + "="*80)
        console.print("[bold green]🎉 Setup Complete! Welcome to YAMLLM![/bold green]")
        console.print("="*80)
        
        success_panel = Panel(
            f"""[bold]Your setup:[/bold]
• Provider: {self.selected_provider.title()}
• Preset: {self.selected_preset}
• Config file: {self.config_filename}

[bold]Ready to start chatting![/bold]""",
            title="Success",
            border_style="green",
            box=box.ROUNDED
        )
        console.print(success_panel)
        
        console.print("\n[bold]Next steps:[/bold]")
        console.print(f"1. Start chatting: [dim]yamllm chat --config {self.config_filename}[/dim]")
        console.print("2. Check status: [dim]yamllm status[/dim]")
        console.print("3. Get help: [dim]yamllm --help[/dim]")
        
        console.print("\n[bold]Tips:[/bold]")
        console.print("• Your API key is set for this session")
        console.print("• Add it to your shell profile or .env file to make it permanent")
        console.print("• Run 'yamllm guide' for more detailed help")
        
        console.print("\n[dim]Happy chatting! 🤖[/dim]")