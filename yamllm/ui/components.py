"""
Enhanced UI components for YAMLLM using Rich.

This module provides beautiful, reusable UI components for displaying
LLM interactions, tool executions, and system status.
"""

from typing import Optional, Dict, Any
import time
from datetime import datetime
from contextlib import contextmanager

from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.live import Live
from rich.text import Text
from rich.markdown import Markdown
from rich.box import ROUNDED, DOUBLE, HEAVY, Box
from rich import box
from rich.tree import Tree
from rich.prompt import Prompt, Confirm


# Custom box style for YAMLLM
YAMLLM_BOX = Box(
    """\
╭─┬╮
│ ││
├─┼┤
│ ││
├─┼┤
├─┼┤
│ ││
╰─┴╯
"""
)


class YAMLLMConsole:
    """Enhanced console with YAMLLM branding and features."""
    
    def __init__(self, theme: str = "default"):
        """Initialize YAMLLM console with theme."""
        # Force terminal colors to avoid environments where Rich cannot auto-detect
        # capabilities (e.g., when output is captured). This keeps the UI beautiful.
        self.console = Console(force_terminal=True, color_system="auto", record=True)
        self.theme = self._load_theme(theme)
        self._start_time = time.time()
    
    def _load_theme(self, theme_name: str) -> Dict[str, Any]:
        """Load color theme."""
        themes = {
            "default": {
                "primary": "cyan",
                "secondary": "magenta",
                "success": "green",
                "warning": "yellow",
                "error": "red",
                "info": "blue",
                "muted": "dim white",
                "highlight": "bold cyan",
                "tool": "yellow",
                "thinking": "dim cyan",
                "user": "green",
                "assistant": "blue",
                "system": "magenta"
            },
            "monokai": {
                "primary": "#66D9EF",
                "secondary": "#F92672",
                "success": "#A6E22E",
                "warning": "#FD971F",
                "error": "#F92672",
                "info": "#66D9EF",
                "muted": "#75715E",
                "highlight": "bold #66D9EF",
                "tool": "#FD971F",
                "thinking": "#75715E",
                "user": "#A6E22E",
                "assistant": "#66D9EF",
                "system": "#F92672"
            },
            "dracula": {
                "primary": "#BD93F9",
                "secondary": "#FF79C6",
                "success": "#50FA7B",
                "warning": "#F1FA8C",
                "error": "#FF5555",
                "info": "#8BE9FD",
                "muted": "#6272A4",
                "highlight": "bold #BD93F9",
                "tool": "#FFB86C",
                "thinking": "#6272A4",
                "user": "#50FA7B",
                "assistant": "#8BE9FD",
                "system": "#FF79C6"
            }
        }
        return themes.get(theme_name, themes["default"])
    
    def print_banner(self):
        """Print YAMLLM banner with ASCII art."""
        banner = """
[bold cyan]
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  ██╗   ██╗ █████╗ ███╗   ███╗██╗     ██╗     ███╗   ███╗                   ║
║  ╚██╗ ██╔╝██╔══██╗████╗ ████║██║     ██║     ████╗ ████║                   ║
║   ╚████╔╝ ███████║██╔████╔██║██║     ██║     ██╔████╔██║                   ║
║    ╚██╔╝  ██╔══██║██║╚██╔╝██║██║     ██║     ██║╚██╔╝██║                   ║
║     ██║   ██║  ██║██║ ╚═╝ ██║███████╗███████╗██║ ╚═╝ ██║                   ║
║     ╚═╝   ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝╚═╝     ╚═╝                   ║
║                                                                              ║
║           [dim]YAML-based Language Model Configuration & Execution[/dim]              ║
║                         [dim]Version 0.1.12 | Rich UI[/dim]                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
[/bold cyan]
"""
        self.console.print(banner)
    
    def print_config_summary(self, config: Dict[str, Any]):
        """Print configuration summary in a beautiful table."""
        table = Table(
            title="[bold]Configuration Summary[/bold]",
            box=ROUNDED,
            show_header=True,
            header_style=f"bold {self.theme['primary']}"
        )
        
        table.add_column("Category", style=self.theme['secondary'])
        table.add_column("Setting", style=self.theme['muted'])
        table.add_column("Value", style=self.theme['info'])
        
        # Provider info
        table.add_row(
            "Provider",
            "Name",
            config.get("provider", {}).get("name", "Unknown")
        )
        table.add_row(
            "",
            "Model",
            config.get("provider", {}).get("model", "Unknown")
        )
        
        # Model settings
        table.add_row(
            "Model Settings",
            "Temperature",
            str(config.get("model_settings", {}).get("temperature", 0.7))
        )
        table.add_row(
            "",
            "Max Tokens",
            str(config.get("model_settings", {}).get("max_tokens", 1000))
        )
        
        # Tools
        tools_enabled = config.get("tools", {}).get("enabled", False)
        table.add_row(
            "Tools",
            "Enabled",
            "✓" if tools_enabled else "✗",
            style=self.theme['success'] if tools_enabled else self.theme['error']
        )
        
        if tools_enabled:
            tools_list = config.get("tools", {}).get("tools", [])
            packs = config.get("tools", {}).get("packs", [])
            table.add_row(
                "",
                "Tools",
                f"{len(tools_list)} tools, {len(packs)} packs"
            )
        
        # Memory
        memory_enabled = config.get("context", {}).get("memory", {}).get("enabled", False)
        table.add_row(
            "Memory",
            "Enabled",
            "✓" if memory_enabled else "✗",
            style=self.theme['success'] if memory_enabled else self.theme['error']
        )
        
        self.console.print(table)
    
    def create_message_panel(
        self,
        role: str,
        content: str,
        timestamp: Optional[datetime] = None,
        tokens: Optional[int] = None,
        thinking: Optional[str] = None
    ) -> Panel:
        """Create a beautiful message panel."""
        # Role styling
        role_colors = {
            "user": self.theme['user'],
            "assistant": self.theme['assistant'],
            "system": self.theme['system'],
            "tool": self.theme['tool'],
            "thinking": self.theme['thinking']
        }
        
        role_icons = {
            "user": "👤",
            "assistant": "🤖",
            "system": "⚙️",
            "tool": "🔧",
            "thinking": "💭"
        }
        
        # Create header
        header = Text()
        header.append(f"{role_icons.get(role, '📝')} ", style="bold")
        header.append(role.title(), style=f"bold {role_colors.get(role, 'white')}")
        
        if timestamp:
            header.append(" • ", style="dim")
            header.append(timestamp.strftime("%H:%M:%S"), style="dim")
        
        if tokens:
            header.append(" • ", style="dim")
            header.append(f"{tokens} tokens", style="dim")
        
        # Create content group
        content_parts = []
        
        # Add thinking section if present
        if thinking:
            thinking_panel = Panel(
                Markdown(thinking),
                title="[bold]Thinking Process[/bold]",
                title_align="left",
                border_style=self.theme['thinking'],
                box=box.MINIMAL
            )
            content_parts.append(thinking_panel)
        
        # Add main content
        if role == "assistant" and "```" in content:
            # Rich markdown rendering for assistant
            content_parts.append(Markdown(content))
        elif role == "tool":
            # Syntax highlighting for tool results
            try:
                import json
                parsed = json.loads(content)
                syntax = Syntax(
                    json.dumps(parsed, indent=2),
                    "json",
                    theme="monokai",
                    line_numbers=False
                )
                content_parts.append(syntax)
            except:
                content_parts.append(Text(content))
        else:
            content_parts.append(Text(content))
        
        # Create panel
        return Panel(
            Group(*content_parts) if len(content_parts) > 1 else content_parts[0],
            title=header,
            title_align="left",
            border_style=role_colors.get(role, "white"),
            box=ROUNDED,
            padding=(1, 2)
        )
    
    def print_message(
        self,
        role: str,
        content: str,
        timestamp: Optional[datetime] = None,
        tokens: Optional[int] = None,
        thinking: Optional[str] = None
    ):
        """Print a message panel."""
        panel = self.create_message_panel(role, content, timestamp, tokens, thinking)
        self.console.print(panel)
    
    @contextmanager
    def live_status(self, message: str):
        """Create a live status indicator."""
        with self.console.status(message, spinner="dots") as status:
            yield status
    
    def create_tool_execution_panel(
        self,
        tool_name: str,
        args: Dict[str, Any],
        result: Any,
        execution_time: float
    ) -> Panel:
        """Create a panel for tool execution display."""
        # Create tree structure
        tree = Tree(f"[bold {self.theme['tool']}]🔧 {tool_name}[/bold {self.theme['tool']}]")
        
        # Add arguments
        args_branch = tree.add("[dim]Arguments[/dim]")
        for key, value in args.items():
            args_branch.add(f"{key}: {value}")
        
        # Add result
        result_branch = tree.add("[dim]Result[/dim]")
        if isinstance(result, dict):
            for key, value in result.items():
                result_branch.add(f"{key}: {value}")
        else:
            result_branch.add(str(result))
        
        # Add execution time
        tree.add(f"[dim]Execution Time: {execution_time:.2f}s[/dim]")
        
        return Panel(
            tree,
            title="[bold]Tool Execution[/bold]",
            border_style=self.theme['tool'],
            box=ROUNDED
        )
    
    def print_tool_execution(
        self,
        tool_name: str,
        args: Dict[str, Any],
        result: Any,
        execution_time: float
    ):
        """Print tool execution panel."""
        panel = self.create_tool_execution_panel(tool_name, args, result, execution_time)
        self.console.print(panel)
    
    def create_progress_bar(self, description: str = "Processing..."):
        """Create a progress bar for long operations."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console
        )
    
    def print_stats(self, stats: Dict[str, Any]):
        """Print statistics in a formatted way."""
        # Create stats table
        table = Table(
            title="[bold]Session Statistics[/bold]",
            box=HEAVY,
            show_header=False,
            padding=(0, 1)
        )
        
        table.add_column("Metric", style=self.theme['secondary'])
        table.add_column("Value", style=self.theme['info'], justify="right")
        
        # Add stats
        if "total_tokens" in stats:
            table.add_row("Total Tokens", f"{stats['total_tokens']:,}")
        if "prompt_tokens" in stats:
            table.add_row("Prompt Tokens", f"{stats['prompt_tokens']:,}")
        if "completion_tokens" in stats:
            table.add_row("Completion Tokens", f"{stats['completion_tokens']:,}")
        if "tool_calls" in stats:
            table.add_row("Tool Calls", str(stats['tool_calls']))
        
        # Add timing
        elapsed = time.time() - self._start_time
        table.add_row("Session Duration", f"{elapsed:.1f}s")
        
        # Create cost estimate if applicable
        if "total_tokens" in stats and stats.get("model"):
            cost = self._estimate_cost(stats["total_tokens"], stats["model"])
            if cost:
                table.add_row("Estimated Cost", f"${cost:.4f}")
        
        self.console.print(table)
    
    def _estimate_cost(self, tokens: int, model: str) -> Optional[float]:
        """Estimate cost based on token usage."""
        # Simple cost estimates (update with actual pricing)
        pricing = {
            "gpt-4": 0.03 / 1000,
            "gpt-3.5-turbo": 0.002 / 1000,
            "claude-3": 0.025 / 1000,
            "gemini-pro": 0.001 / 1000
        }
        
        for model_key, price in pricing.items():
            if model_key in model.lower():
                return tokens * price
        
        return None
    
    def print_error(self, error: Exception, details: Optional[Dict[str, Any]] = None):
        """Print error in a formatted panel."""
        # Create error content
        error_text = Text()
        error_text.append("❌ ", style="bold red")
        error_text.append(type(error).__name__, style="bold red")
        error_text.append(": ", style="red")
        error_text.append(str(error), style="red")
        
        # Add details if provided
        content_parts = [error_text]
        
        if details:
            detail_tree = Tree("[dim]Error Details[/dim]")
            for key, value in details.items():
                detail_tree.add(f"{key}: {value}")
            content_parts.append(detail_tree)
        
        # Create panel
        panel = Panel(
            Group(*content_parts),
            title="[bold red]Error[/bold red]",
            border_style="red",
            box=DOUBLE
        )
        
        self.console.print(panel)
    
    def prompt_user(self, prompt: str = "You", multiline: bool = False) -> str:
        """Get user input with nice formatting."""
        if multiline:
            self.console.print(
                f"[{self.theme['user']}]📝 {prompt} (Ctrl+D to submit):[/{self.theme['user']}]"
            )
            lines = []
            while True:
                try:
                    line = Prompt.ask("", console=self.console)
                    lines.append(line)
                except EOFError:
                    break
            return "\n".join(lines)
        else:
            return Prompt.ask(
                f"[{self.theme['user']}]💬 {prompt}[/{self.theme['user']}]",
                console=self.console
            )
    
    def confirm(self, message: str) -> bool:
        """Get user confirmation."""
        return Confirm.ask(
            f"[{self.theme['warning']}]{message}[/{self.theme['warning']}]",
            console=self.console
        )


class StreamingDisplay:
    """Handle streaming display with Rich."""
    
    def __init__(self, console: YAMLLMConsole):
        self.console = console
        self.current_text = ""
        self.panel = None
        self.live = None
    
    def start(self, role: str = "assistant"):
        """Start streaming display."""
        self.current_text = ""
        self.role = role
        self.start_time = datetime.now()
        
        # Create initial empty panel
        self.panel = self.console.create_message_panel(
            role=role,
            content="",
            timestamp=self.start_time
        )
        
        # Start live display
        self.live = Live(self.panel, console=self.console.console, refresh_per_second=10)
        self.live.start()
    
    def update(self, text: str):
        """Update streaming display with new text."""
        self.current_text += text
        
        # Update panel
        self.panel = self.console.create_message_panel(
            role=self.role,
            content=self.current_text,
            timestamp=self.start_time,
            tokens=len(self.current_text.split())  # Rough estimate
        )
        
        self.live.update(self.panel)
    
    def stop(self):
        """Stop streaming display."""
        if self.live:
            self.live.stop()
            self.live = None


class ToolExecutionDisplay:
    """Display tool executions with animations."""
    
    def __init__(self, console: YAMLLMConsole):
        self.console = console
        self.active_tools = {}
    
    @contextmanager
    def track_tool(self, tool_name: str, args: Dict[str, Any]):
        """Track tool execution with animation."""
        # Show tool starting
        with self.console.console.status(
            f"[{self.console.theme['tool']}]🔧 Executing {tool_name}...[/{self.console.theme['tool']}]",
            spinner="dots"
        ):
            start_time = time.time()
            yield
            execution_time = time.time() - start_time
        
        # Tool completed - this will be called by the main app
        # when it has the result
        self.execution_time = execution_time
    
    def show_result(self, tool_name: str, args: Dict[str, Any], result: Any):
        """Show tool execution result."""
        self.console.print_tool_execution(
            tool_name, args, result, getattr(self, 'execution_time', 0.0)
        )
