"""
Optional Rich-based streaming renderer for LLM.

Usage:
    from yamllm.ui.rich_renderer import attach_stream_renderer
    attach_stream_renderer(llm)  # installs a callback to render streamed deltas
"""

from typing import Optional


def attach_stream_renderer(llm, console: Optional[object] = None) -> None:
    try:
        from rich.console import Console
    except Exception:  # pragma: no cover - Rich not installed
        def _printer(delta: str) -> None:
            print(delta, end="", flush=True)
        llm.set_stream_callback(_printer)
        return

    cons = console or Console()

    def _on_delta(delta: str) -> None:
        # For simplicity, print raw deltas; caller can customize
        cons.print(delta, end="")

    llm.set_stream_callback(_on_delta)

