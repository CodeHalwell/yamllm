"""UI components for agent progress visualization."""

from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from rich.tree import Tree

from yamllm.agent.models import AgentState, Task, TaskStatus
from yamllm.ui.themes import theme_manager


class AgentUI:
    """UI for displaying agent progress and state."""

    def __init__(self, console: Optional[Console] = None):
        """Initialize agent UI."""
        self.console = console or Console()
        self.theme = theme_manager.current_theme

    def render_header(self, state: AgentState) -> None:
        """Render agent header with goal."""
        header = Panel(
            f"[bold]{state.goal}[/bold]",
            title="[bold cyan]Agent Goal[/bold cyan]",
            border_style=self.theme.colors.primary,
            box=box.ROUNDED
        )
        self.console.print(header)

    def render_progress(self, state: AgentState) -> None:
        """Render current progress summary."""
        progress_pct = state.get_progress()
        completed = len(state.get_completed_tasks())
        total = len(state.tasks)

        # Create progress text
        text = Text()
        text.append(f"Iteration: ", style="dim")
        text.append(f"{state.iteration}/{state.max_iterations}", style=self.theme.colors.info)
        text.append(" | ", style="dim")
        text.append(f"Progress: ", style="dim")
        text.append(f"{progress_pct:.0f}%", style=self.theme.colors.success)
        text.append(f" ({completed}/{total} tasks)", style="dim")

        self.console.print(text)

    def render_task_tree(self, state: AgentState) -> None:
        """Render task dependency tree with progress."""
        tree = Tree(
            "📋 [bold]Tasks[/bold]",
            guide_style=self.theme.colors.dim
        )

        for task in state.tasks:
            # Choose icon and style based on status
            icon, style = self._get_task_display(task)

            # Create task node
            task_text = f"{icon} [{style}]{task.id}: {task.description}[/{style}]"
            node = tree.add(task_text)

            # Add metadata if available
            if task.metadata:
                complexity = task.metadata.get("complexity")
                if complexity:
                    node.add(f"[dim]Complexity: {complexity}[/dim]")

            # Add result if completed
            if task.status == TaskStatus.COMPLETED and task.result:
                result = task.result
                response = result.get("response", "")
                if response:
                    # Truncate long responses
                    truncated = response[:100] + "..." if len(response) > 100 else response
                    node.add(f"[dim]{truncated}[/dim]")

            # Add error if failed
            if task.status == TaskStatus.FAILED and task.error:
                node.add(f"[red]Error: {task.error}[/red]")

        self.console.print(Panel(tree, border_style=self.theme.colors.info, box=box.ROUNDED))

    def render_task_list(self, state: AgentState) -> None:
        """Render simple task list."""
        table = Table(
            title="Tasks",
            box=box.SIMPLE,
            show_header=True,
            header_style=f"bold {self.theme.colors.primary}"
        )

        table.add_column("ID", style="cyan", width=10)
        table.add_column("Description", style="white")
        table.add_column("Status", width=12)

        for task in state.tasks:
            icon, style = self._get_task_display(task)
            status_text = f"{icon} [{style}]{task.status.value.upper()}[/{style}]"

            table.add_row(
                task.id,
                task.description[:60] + "..." if len(task.description) > 60 else task.description,
                status_text
            )

        self.console.print(table)

    def render_current_thought(self, state: AgentState) -> None:
        """Render agent's current thought."""
        if not state.thought_history:
            return

        latest_thought = state.thought_history[-1]

        panel = Panel(
            f"[italic]{latest_thought}[/italic]",
            title="💭 [bold]Current Thought[/bold]",
            border_style=self.theme.colors.secondary,
            box=box.ROUNDED
        )

        self.console.print(panel)

    def render_current_action(self, task: Task) -> None:
        """Render current action being executed."""
        text = Text()
        text.append("⚡ ", style=self.theme.colors.warning)
        text.append("Executing: ", style="bold")
        text.append(task.description, style=self.theme.colors.info)

        self.console.print(text)

    def render_full_state(self, state: AgentState) -> None:
        """Render complete agent state."""
        self.console.clear()
        self.render_header(state)
        self.console.print()
        self.render_progress(state)
        self.console.print()
        self.render_task_tree(state)
        self.console.print()

        if state.current_task_id:
            current_task = state.get_task_by_id(state.current_task_id)
            if current_task:
                self.render_current_action(current_task)
                self.console.print()

        self.render_current_thought(state)

    def render_completion(self, state: AgentState) -> None:
        """Render completion summary."""
        if state.success:
            title = "✅ [bold green]Goal Achieved![/bold green]"
            style = "green"
        else:
            title = "❌ [bold red]Goal Not Achieved[/bold red]"
            style = "red"

        # Create summary
        completed = len(state.get_completed_tasks())
        failed = len([t for t in state.tasks if t.status == TaskStatus.FAILED])

        summary_text = f"""
Iterations: {state.iteration}/{state.max_iterations}
Tasks Completed: {completed}/{len(state.tasks)}
Tasks Failed: {failed}
"""

        if state.error:
            summary_text += f"\nError: {state.error}"

        panel = Panel(
            summary_text.strip(),
            title=title,
            border_style=style,
            box=box.DOUBLE
        )

        self.console.print(panel)

        # Show learnings if any
        if "learnings" in state.metadata and state.metadata["learnings"]:
            self.console.print("\n[bold]Key Learnings:[/bold]")
            for i, learning in enumerate(state.metadata["learnings"], 1):
                self.console.print(f"  {i}. {learning}")

    def stream_execution(self, state: AgentState) -> Live:
        """Create a Live display for streaming execution."""
        return Live(
            self._generate_display(state),
            console=self.console,
            refresh_per_second=4,
            vertical_overflow="visible"
        )

    def _generate_display(self, state: AgentState) -> Panel:
        """Generate display panel for live updates."""
        # Create content
        content_parts = []

        # Progress
        progress_pct = state.get_progress()
        completed = len(state.get_completed_tasks())
        total = len(state.tasks)

        progress_text = Text()
        progress_text.append(f"Iteration: {state.iteration}/{state.max_iterations} | ", style="dim")
        progress_text.append(f"Progress: {progress_pct:.0f}% ", style=self.theme.colors.success)
        progress_text.append(f"({completed}/{total})", style="dim")
        content_parts.append(progress_text)

        # Current task
        if state.current_task_id:
            task = state.get_task_by_id(state.current_task_id)
            if task:
                task_text = Text()
                task_text.append("\n▶ ", style=self.theme.colors.warning)
                task_text.append(f"{task.description}", style="bold")
                content_parts.append(task_text)

        # Recent thought
        if state.thought_history:
            thought = state.thought_history[-1]
            thought_text = Text()
            thought_text.append("\n💭 ", style=self.theme.colors.secondary)
            thought_text.append(thought[:150] + "..." if len(thought) > 150 else thought, style="italic dim")
            content_parts.append(thought_text)

        # Combine parts
        content = Text()
        for part in content_parts:
            content.append(part)

        return Panel(
            content,
            title=f"[bold cyan]Agent: {state.goal[:50]}...[/bold cyan]",
            border_style=self.theme.colors.primary,
            box=box.ROUNDED
        )

    def _get_task_display(self, task: Task) -> tuple:
        """Get display icon and style for task."""
        if task.status == TaskStatus.COMPLETED:
            return "✓", self.theme.colors.success
        elif task.status == TaskStatus.FAILED:
            return "✗", self.theme.colors.error
        elif task.status == TaskStatus.IN_PROGRESS:
            return "▶", self.theme.colors.warning
        elif task.status == TaskStatus.BLOCKED:
            return "⊗", self.theme.colors.dim
        else:  # PENDING
            return "⋯", self.theme.colors.dim
