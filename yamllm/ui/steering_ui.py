"""Rich UI components for interactive agent steering."""

from typing import Optional, Dict, Any, List
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.layout import Layout
from rich.text import Text
from rich import box


class SteeringUI:
    """Rich UI for interactive agent steering."""

    def __init__(self, console: Optional[Console] = None):
        """Initialize steering UI."""
        self.console = console or Console()

    def render_agent_dashboard(
        self,
        state: Any,
        current_thought: str,
        planned_action: Dict[str, Any],
        decision_history: List[Any]
    ):
        """
        Render complete agent dashboard.

        Args:
            state: Current agent state
            current_thought: Current reasoning
            planned_action: Planned next action
            decision_history: History of steering decisions
        """
        layout = Layout()

        # Split into sections
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=8)
        )

        # Split main into left and right
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )

        # Header
        header_text = Text("Agent Steering Dashboard", style="bold cyan", justify="center")
        layout["header"].update(Panel(header_text, border_style="cyan"))

        # Left: Current state
        layout["left"].update(self._render_current_state(state, current_thought, planned_action))

        # Right: History and context
        layout["right"].update(self._render_history(decision_history))

        # Footer: Controls
        layout["footer"].update(self._render_controls())

        self.console.print(layout)

    def _render_current_state(
        self,
        state: Any,
        thought: str,
        action: Dict[str, Any]
    ) -> Panel:
        """Render current agent state panel."""
        content = []

        # Progress
        if hasattr(state, 'get_progress'):
            progress = state.get_progress()
            completed = len(state.get_completed_tasks()) if hasattr(state, 'get_completed_tasks') else 0
            total = len(state.tasks) if hasattr(state, 'tasks') else 0

            content.append(f"[bold]Progress:[/bold] {progress:.1%} ({completed}/{total} tasks)")

        # Current thought
        content.append(f"\n[bold yellow]Reasoning:[/bold yellow]\n{thought[:200]}")

        # Planned action
        content.append(f"\n[bold green]Next Action:[/bold green]")
        for key, value in list(action.items())[:3]:
            content.append(f"  {key}: {str(value)[:50]}")

        return Panel(
            "\n".join(content),
            title="Current State",
            border_style="blue"
        )

    def _render_history(self, decision_history: List[Any]) -> Panel:
        """Render decision history panel."""
        table = Table(box=box.SIMPLE)
        table.add_column("#", style="dim", width=4)
        table.add_column("Action", style="cyan")
        table.add_column("Details", style="white")

        for i, decision in enumerate(decision_history[-10:], 1):  # Last 10
            action_style = {
                "approve": "green",
                "reject": "red",
                "modify": "yellow",
                "skip": "blue"
            }.get(decision.action.value, "white")

            details = decision.feedback[:30] if decision.feedback else "-"

            table.add_row(
                str(i),
                f"[{action_style}]{decision.action.value}[/{action_style}]",
                details
            )

        return Panel(table, title="Decision History", border_style="magenta")

    def _render_controls(self) -> Panel:
        """Render control panel."""
        controls = [
            "[green]a[/green] Approve  [yellow]m[/yellow] Modify  [red]r[/red] Reject",
            "[blue]p[/blue] Pause     [magenta]s[/magenta] Skip    [red]x[/red] Stop",
            "[cyan]auto[/cyan] Auto-approve remaining"
        ]

        return Panel(
            "\n".join(controls),
            title="Controls",
            border_style="white"
        )

    def render_task_tree(self, state: Any):
        """Render task tree showing dependencies."""
        tree = Tree("[bold]Agent Tasks[/bold]")

        if not hasattr(state, 'tasks'):
            return

        # Group by status
        from ..models import TaskStatus

        pending = [t for t in state.tasks if t.status == TaskStatus.PENDING]
        in_progress = [t for t in state.tasks if t.status == TaskStatus.IN_PROGRESS]
        completed = [t for t in state.tasks if t.status == TaskStatus.COMPLETED]
        failed = [t for t in state.tasks if t.status == TaskStatus.FAILED]

        if in_progress:
            progress_branch = tree.add("[yellow]⏳ In Progress[/yellow]")
            for task in in_progress:
                progress_branch.add(f"[yellow]{task.description}[/yellow]")

        if completed:
            completed_branch = tree.add("[green]✓ Completed[/green]")
            for task in completed[:5]:  # Limit display
                completed_branch.add(f"[green]{task.description}[/green]")

        if pending:
            pending_branch = tree.add("[blue]○ Pending[/blue]")
            for task in pending[:5]:  # Limit display
                pending_branch.add(f"[blue]{task.description}[/blue]")

        if failed:
            failed_branch = tree.add("[red]✗ Failed[/red]")
            for task in failed:
                failed_branch.add(f"[red]{task.description}[/red]")

        self.console.print(Panel(tree, title="Task Overview", border_style="cyan"))

    def render_watchpoints(self, watchpoints: List[str]):
        """Render active watchpoints."""
        if not watchpoints:
            return

        table = Table(title="Active Watchpoints", box=box.ROUNDED)
        table.add_column("#", style="cyan", width=4)
        table.add_column("Condition", style="yellow")

        for i, wp in enumerate(watchpoints, 1):
            table.add_row(str(i), wp)

        self.console.print(table)

    def show_progress(self, state: Any):
        """Show animated progress bar."""
        if not hasattr(state, 'get_progress'):
            return

        progress = state.get_progress()
        completed = len(state.get_completed_tasks()) if hasattr(state, 'get_completed_tasks') else 0
        total = len(state.tasks) if hasattr(state, 'tasks') else 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console
        ) as progress_bar:
            task = progress_bar.add_task(
                f"Agent Progress ({completed}/{total} tasks)",
                total=100
            )
            progress_bar.update(task, completed=progress * 100)

    def render_alternatives(self, alternatives: List[Dict[str, Any]]):
        """Render suggested alternative actions."""
        if not alternatives:
            return

        self.console.print("\n[bold cyan]Suggested Alternatives:[/bold cyan]\n")

        table = Table(box=box.ROUNDED)
        table.add_column("#", style="cyan", width=4)
        table.add_column("Alternative", style="yellow")
        table.add_column("Reason", style="white")

        for i, alt in enumerate(alternatives[:5], 1):
            table.add_row(
                str(i),
                alt.get("action", "Unknown"),
                alt.get("reason", "")[:50]
            )

        self.console.print(table)

    def show_comparison(
        self,
        original_action: Dict[str, Any],
        modified_action: Dict[str, Any]
    ):
        """Show comparison between original and modified actions."""
        self.console.print("\n[bold]Action Comparison:[/bold]\n")

        table = Table(box=box.ROUNDED)
        table.add_column("Property", style="cyan")
        table.add_column("Original", style="yellow")
        table.add_column("Modified", style="green")

        # Find differences
        all_keys = set(original_action.keys()) | set(modified_action.keys())

        for key in all_keys:
            orig_val = str(original_action.get(key, "-"))[:40]
            mod_val = str(modified_action.get(key, "-"))[:40]

            # Highlight differences
            style = "bold" if orig_val != mod_val else ""

            table.add_row(
                key,
                f"[{style}]{orig_val}[/{style}]",
                f"[{style}]{mod_val}[/{style}]"
            )

        self.console.print(table)

    def render_state_inspector(self, state: Any):
        """Render detailed state inspector."""
        self.console.print("\n[bold cyan]═══ State Inspector ═══[/bold cyan]\n")

        if hasattr(state, '__dict__'):
            table = Table(title="Agent State", box=box.DOUBLE)
            table.add_column("Property", style="cyan", width=20)
            table.add_column("Value", style="white")

            for key, value in state.__dict__.items():
                if not key.startswith('_'):
                    # Format complex values
                    if isinstance(value, (list, dict)):
                        val_str = f"{type(value).__name__}[{len(value)}]"
                    else:
                        val_str = str(value)[:100]

                    table.add_row(key, val_str)

            self.console.print(table)

    def show_error(self, error_msg: str, context: Optional[Dict[str, Any]] = None):
        """Show error message with context."""
        content = [f"[bold red]Error:[/bold red] {error_msg}"]

        if context:
            content.append("\n[bold]Context:[/bold]")
            for key, value in context.items():
                content.append(f"  {key}: {value}")

        self.console.print(Panel(
            "\n".join(content),
            title="Error",
            border_style="red"
        ))

    def show_success(self, message: str):
        """Show success message."""
        self.console.print(Panel(
            f"[bold green]✓ {message}[/bold green]",
            border_style="green"
        ))

    def show_warning(self, message: str):
        """Show warning message."""
        self.console.print(Panel(
            f"[bold yellow]⚠ {message}[/bold yellow]",
            border_style="yellow"
        ))

    def render_diff(self, before: str, after: str, context_lines: int = 3):
        """Render diff between before and after states."""
        try:
            import difflib

            diff = difflib.unified_diff(
                before.splitlines(),
                after.splitlines(),
                lineterm='',
                n=context_lines
            )

            diff_text = "\n".join(diff)

            if diff_text:
                from rich.syntax import Syntax
                syntax = Syntax(diff_text, "diff", theme="monokai", line_numbers=True)
                self.console.print(Panel(syntax, title="Changes", border_style="yellow"))

        except Exception as e:
            self.console.print(f"[dim]Could not render diff: {e}[/dim]")

    def clear(self):
        """Clear console."""
        self.console.clear()

    def pause(self, message: str = "Press Enter to continue..."):
        """Pause and wait for user input."""
        self.console.input(f"\n[dim]{message}[/dim]")
