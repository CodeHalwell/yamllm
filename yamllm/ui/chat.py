"""
Rich-based chat UI helpers for LLM streaming.
"""

from typing import Optional
import os
import re

from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.live import Live
from rich.markdown import Markdown
import json

from yamllm.ui.themes import theme_manager, Theme

class TerminalUI:
    """A Rich-based terminal UI for chat applications."""

    def __init__(self, style: str = "bubble", console: Optional[Console] = None):
        self.style = os.getenv("YAMLLM_CHAT_STYLE", style)
        self.console = console or Console()
        self.theme: Theme = theme_manager.current_theme

    def set_theme(self, theme_name: str):
        theme_manager.set_theme(theme_name)
        self.theme = theme_manager.current_theme

    def print_header(self, provider: str, model: str):
        """Prints the header for the chat session."""
        if self.theme.layout.show_ascii_art:
            art, _ = _provider_art(provider)
            if art:
                self.console.print(Panel(Text(art, style=f"{self.theme.colors.primary} italic"), box=box.ROUNDED, border_style=self.theme.colors.primary))
        
        title = Text("yamllm", style="bold white").append(" • ")
        title.append(provider.capitalize(), style=self.theme.colors.primary)
        subtitle = Text("model: ", style="dim").append(model, style=self.theme.colors.secondary)
        self.console.print(Panel(title.append("  ").append(subtitle), box=box.ROUNDED, border_style="grey39"))

    def print_user_message(self, content: str):
        """Renders a user message."""
        if self.style == "compact":
            self.console.print(Text("You ", style=f"bold {self.theme.colors.primary}").append(content))
            return
        panel = Panel(
            Text(content, style="white"),
            title="You",
            title_align="left",
            box=box.ROUNDED if self.style == "bubble" else box.MINIMAL,
            border_style=self.theme.colors.primary,
        )
        self.console.print(panel)

    def print_assistant_message(self, content: str):
        """Renders a complete assistant message."""
        self.console.print(self._assistant_panel(Markdown(content)))

    def print_event(self, ev: dict):
        """Renders an event, like a tool call or result."""
        try:
            et = ev.get("type")
            color = self.theme.colors.secondary
            if et == "tool_request":
                names = ", ".join([c.get("function", {}).get("name") for c in ev.get("tool_calls", []) if c])
                panel = Panel(Text(f"Model requested tools: {names}", style=color), title="Tool Request", border_style=color, box=box.ROUNDED)
                self.console.print(panel)
            elif et == "tool_call":
                name = ev.get("name")
                args = ev.get("args", {})
                preview = Text(json.dumps(args), style=color) if args else Text("(no args)", style=color)
                panel = Panel(preview, title=f"Calling: {name}", border_style=color, box=box.ROUNDED)
                self.console.print(panel)
            elif et == "tool_result":
                name = ev.get("name")
                res = ev.get("result")
                try:
                    rendered = Markdown(res) if isinstance(res, str) else Text(json.dumps(res, indent=2))
                except Exception:
                    rendered = Text(str(res))
                panel = Panel(rendered, title=f"Result: {name}", border_style=color, box=box.ROUNDED)
                self.console.print(panel)
            elif et == "model_usage":
                u = ev.get("usage", {})
                info = Text(f"Tokens — prompt: {u.get('prompt_tokens', 0)}, completion: {u.get('completion_tokens', 0)}, total: {u.get('total_tokens', 0)}", style=self.theme.colors.dim)
                self.console.print(Panel(info, title="Usage", border_style="grey39", box=box.MINIMAL))
        except Exception:
            pass

    def stream_assistant_response(self, llm, prompt: str) -> str:
        """Streams an assistant's response to the terminal."""
        reply_buffer = ""
        full_response = Text()
        
        with Live(self._assistant_panel(full_response), console=self.console, refresh_per_second=10, vertical_overflow="visible") as live:
            def on_delta(delta: str) -> None:
                nonlocal reply_buffer
                nonlocal full_response
                
                reply_buffer += delta
                
                # Process buffer for complete sentences or newlines
                while True:
                    match = re.search(r"([.!?\n])", reply_buffer)
                    if match:
                        split_pos = match.end()
                        sentence = reply_buffer[:split_pos]
                        reply_buffer = reply_buffer[split_pos:]
                        
                        full_response.append(sentence)
                        live.update(self._assistant_panel(Markdown(full_response.plain)))
                    else:
                        break

            llm.set_stream_callback(on_delta)
            llm.set_event_callback(self.print_event)
            
            try:
                result = llm.query(prompt)
                if result:  # non-streaming fallback
                    md = Markdown(result)
                    live.update(self._assistant_panel(md))
                    return result
            finally:
                # Append any remaining text in the buffer
                if reply_buffer:
                    full_response.append(reply_buffer)
                    live.update(self._assistant_panel(Markdown(full_response.plain)))
                
                llm.clear_stream_callback()
                llm.clear_event_callback()
                
        return full_response.plain

    def _assistant_panel(self, renderable: RenderableType) -> Panel:
        return Panel(
            renderable,
            title="Assistant",
            title_align="left",
            box=box.ROUNDED if self.style == "bubble" else box.MINIMAL,
            border_style=self.theme.colors.success,
        )

def _provider_art(name: str):
    name = (name or "").lower()
    arts = {
        "openai": (
            "\n   ╭─────────────╮\n   │  ◈ OPENAI ◈ │\n   ╰─────────────╯\n",
            "bright_green",
        ),
        "google": (
            "\n   ╭─────────────╮\n   │  ◎ GOOGLE ◎ │\n   ╰─────────────╯\n",
            "bright_yellow",
        ),
        "mistral": (
            "\n   ╭───────────────╮\n   │  ~ MISTRAL ~  │\n   ╰───────────────╯\n",
            "bright_cyan",
        ),
        "anthropic": (
            "\n   ╭──────────────────╮\n   │  { ANTHROPIC }   │\n   ╰──────────────────╯\n",
            "magenta",
        ),
        "azure_openai": (
            "\n   ╭─────────────────────╮\n   │  ◆ AZURE OPENAI ◆  │\n   ╰─────────────────────╯\n",
            "blue",
        ),
        "azure_foundry": (
            "\n   ╭───────────────────────╮\n   │  ◆ AZURE FOUNDRY ◆   │\n   ╰───────────────────────╯\n",
            "bright_blue",
        ),
        "openrouter": (
            "\n   ╭──────────────────────╮\n   │  ▶ OPENROUTER ◀  │\n   ╰──────────────────────╯\n",
            "bright_green",
        ),
        "deepseek": (
            "\n   ╭──────────────────╮\n   │  ≋ DEEPSEEK ≋   │\n   ╰──────────────────╯\n",
            "bright_magenta",
        ),
    }
    return arts.get(name, (None, None))