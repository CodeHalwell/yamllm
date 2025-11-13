"""Interactive agent steering with human-in-the-loop control."""

import logging
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass
from enum import Enum
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.syntax import Syntax


class SteeringAction(Enum):
    """Actions available during agent steering."""
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    PAUSE = "pause"
    SKIP = "skip"
    STOP = "stop"
    AUTO = "auto"  # Auto-approve remaining


@dataclass
class SteeringDecision:
    """Represents a human decision during agent execution."""
    action: SteeringAction
    feedback: Optional[str] = None
    modified_task: Optional[str] = None
    auto_approve_remaining: bool = False


@dataclass
class SteeringPoint:
    """A point where agent pauses for human input."""
    iteration: int
    thought: str
    planned_action: Dict[str, Any]
    current_state: Any
    context: Dict[str, Any]


class InteractiveSteering:
    """
    Interactive steering controller for agents.

    Allows human-in-the-loop control with pause, review, and override capabilities.
    """

    def __init__(
        self,
        console: Optional[Console] = None,
        auto_approve: bool = False,
        pause_before_action: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize interactive steering.

        Args:
            console: Rich console for UI (creates new if None)
            auto_approve: Whether to auto-approve all actions
            pause_before_action: Whether to pause before each action
            logger: Optional logger
        """
        self.console = console or Console()
        self.auto_approve = auto_approve
        self.pause_before_action = pause_before_action
        self.logger = logger or logging.getLogger(__name__)

        # Steering state
        self.is_paused = False
        self.should_stop = False
        self.watchpoints: List[Callable[[SteeringPoint], bool]] = []
        self.decision_history: List[SteeringDecision] = []

    def add_watchpoint(self, condition: Callable[[SteeringPoint], bool]):
        """
        Add a watchpoint that triggers pause when condition is met.

        Args:
            condition: Function that takes SteeringPoint and returns bool

        Example:
            steering.add_watchpoint(lambda sp: "delete" in sp.thought.lower())
        """
        self.watchpoints.append(condition)

    def clear_watchpoints(self):
        """Clear all watchpoints."""
        self.watchpoints.clear()

    def check_watchpoints(self, point: SteeringPoint) -> bool:
        """Check if any watchpoint triggers for this steering point."""
        for watchpoint in self.watchpoints:
            try:
                if watchpoint(point):
                    return True
            except Exception as e:
                self.logger.warning(f"Watchpoint check failed: {e}")
        return False

    def request_decision(self, point: SteeringPoint) -> SteeringDecision:
        """
        Request human decision at a steering point.

        Args:
            point: Current steering point

        Returns:
            SteeringDecision from human
        """
        # Check if we should pause
        should_pause = (
            self.pause_before_action
            or self.is_paused
            or self.check_watchpoints(point)
        )

        if self.auto_approve and not should_pause:
            return SteeringDecision(action=SteeringAction.APPROVE)

        # Display steering point
        self._display_steering_point(point)

        # Get user decision
        decision = self._get_user_decision(point)

        # Update state based on decision
        if decision.action == SteeringAction.AUTO:
            self.auto_approve = True
            decision.auto_approve_remaining = True
        elif decision.action == SteeringAction.PAUSE:
            self.is_paused = True
        elif decision.action == SteeringAction.STOP:
            self.should_stop = True

        # Record decision
        self.decision_history.append(decision)

        return decision

    def _display_steering_point(self, point: SteeringPoint):
        """Display current steering point to user."""
        self.console.print(f"\n[bold cyan]╔{'═' * 78}╗[/bold cyan]")
        self.console.print(f"[bold cyan]║ Iteration {point.iteration:2d} - Agent Steering Point{' ' * 44}║[/bold cyan]")
        self.console.print(f"[bold cyan]╚{'═' * 78}╝[/bold cyan]\n")

        # Show thought/reasoning
        self.console.print(Panel(
            point.thought,
            title="[bold yellow]Agent Reasoning[/bold yellow]",
            border_style="yellow"
        ))

        # Show planned action
        action_text = self._format_action(point.planned_action)
        self.console.print(Panel(
            action_text,
            title="[bold green]Planned Action[/bold green]",
            border_style="green"
        ))

        # Show context if available
        if point.context:
            context_items = []
            for key, value in point.context.items():
                context_items.append(f"[cyan]{key}:[/cyan] {value}")

            if context_items:
                self.console.print(Panel(
                    "\n".join(context_items[:5]),  # Show top 5
                    title="[bold blue]Context[/bold blue]",
                    border_style="blue"
                ))

    def _format_action(self, action: Dict[str, Any]) -> str:
        """Format action for display."""
        if not action:
            return "[dim]No action planned[/dim]"

        parts = []
        for key, value in action.items():
            if isinstance(value, str) and len(value) > 100:
                value = value[:100] + "..."
            parts.append(f"[bold]{key}:[/bold] {value}")

        return "\n".join(parts)

    def _get_user_decision(self, point: SteeringPoint) -> SteeringDecision:
        """Get decision from user via interactive prompt."""
        self.console.print("\n[bold]Choose an action:[/bold]")
        self.console.print("  [green]a[/green] - Approve and continue")
        self.console.print("  [yellow]m[/yellow] - Modify action")
        self.console.print("  [red]r[/red] - Reject and skip")
        self.console.print("  [blue]p[/blue] - Pause (review state)")
        self.console.print("  [magenta]s[/magenta] - Skip this iteration")
        self.console.print("  [red]x[/red] - Stop agent execution")
        self.console.print("  [cyan]auto[/cyan] - Auto-approve remaining")

        while True:
            choice = Prompt.ask(
                "\n[bold]Decision[/bold]",
                choices=["a", "m", "r", "p", "s", "x", "auto"],
                default="a"
            )

            if choice == "a":
                return SteeringDecision(action=SteeringAction.APPROVE)

            elif choice == "m":
                feedback = Prompt.ask("[yellow]Modification guidance[/yellow]")
                return SteeringDecision(
                    action=SteeringAction.MODIFY,
                    feedback=feedback
                )

            elif choice == "r":
                feedback = Prompt.ask("[red]Rejection reason (optional)[/red]", default="")
                return SteeringDecision(
                    action=SteeringAction.REJECT,
                    feedback=feedback if feedback else None
                )

            elif choice == "p":
                self._display_full_state(point)
                continue  # Re-prompt

            elif choice == "s":
                return SteeringDecision(action=SteeringAction.SKIP)

            elif choice == "x":
                if Confirm.ask("[red]Stop agent execution?[/red]"):
                    return SteeringDecision(action=SteeringAction.STOP)
                continue

            elif choice == "auto":
                if Confirm.ask("[cyan]Auto-approve all remaining actions?[/cyan]"):
                    return SteeringDecision(action=SteeringAction.AUTO)
                continue

    def _display_full_state(self, point: SteeringPoint):
        """Display complete agent state."""
        self.console.print("\n[bold cyan]═══ Full Agent State ═══[/bold cyan]\n")

        # Show current state
        if hasattr(point.current_state, '__dict__'):
            state_dict = point.current_state.__dict__

            table = Table(title="Agent State")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="white")

            for key, value in state_dict.items():
                if not key.startswith('_'):
                    table.add_row(key, str(value)[:100])

            self.console.print(table)

        # Show all context
        if point.context:
            self.console.print("\n[bold]Context:[/bold]")
            for key, value in point.context.items():
                self.console.print(f"  [cyan]{key}:[/cyan] {value}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of steering session."""
        action_counts = {}
        for decision in self.decision_history:
            action = decision.action.value
            action_counts[action] = action_counts.get(action, 0) + 1

        return {
            "total_decisions": len(self.decision_history),
            "action_counts": action_counts,
            "auto_approved": self.auto_approve,
            "stopped": self.should_stop
        }


class InteractiveAgent:
    """
    Agent with interactive steering capabilities.

    Wraps a standard Agent with interactive control features.
    """

    def __init__(
        self,
        agent,
        steering: Optional[InteractiveSteering] = None,
        pause_before_action: bool = True,
        auto_approve: bool = False
    ):
        """
        Initialize interactive agent.

        Args:
            agent: Base Agent instance
            steering: Optional InteractiveSteering controller
            pause_before_action: Whether to pause before each action
            auto_approve: Whether to auto-approve all actions
        """
        self.agent = agent
        self.steering = steering or InteractiveSteering(
            pause_before_action=pause_before_action,
            auto_approve=auto_approve
        )

    def execute(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Execute agent with interactive steering.

        Args:
            goal: Agent goal
            context: Optional context

        Returns:
            Final agent state
        """
        from ..models import AgentState

        # Initialize state
        state = AgentState(
            goal=goal,
            tasks=[],
            max_iterations=self.agent.max_iterations,
            metadata=context or {}
        )

        # Planning phase
        if self.agent.enable_planning:
            state = self.agent.planner.decompose_goal(goal, context, state)

        # Interactive execution loop
        while not state.completed and state.iteration < state.max_iterations:
            state.iteration += 1

            # Check if we should stop
            if self.steering.should_stop:
                self.steering.console.print("[red]Execution stopped by user[/red]")
                break

            # Reason about next action
            thought, next_task = self.agent.reasoner.reason(state)

            if next_task is None:
                break

            # Create steering point
            point = SteeringPoint(
                iteration=state.iteration,
                thought=thought,
                planned_action={
                    "task_id": next_task.id,
                    "description": next_task.description,
                    "dependencies": next_task.dependencies
                },
                current_state=state,
                context={
                    "completed_tasks": len(state.get_completed_tasks()),
                    "progress": f"{state.get_progress():.1%}",
                    "current_task": next_task.description
                }
            )

            # Request decision
            decision = self.steering.request_decision(point)

            # Handle decision
            if decision.action == SteeringAction.APPROVE:
                # Execute action
                action_result = self.agent.actor.act(next_task, state)
                state.add_action(action_result.to_dict())
                state = self.agent.observer.observe(action_result, state)

            elif decision.action == SteeringAction.MODIFY:
                # Apply modification
                if decision.feedback:
                    # Add feedback as context for next iteration
                    state.metadata["user_feedback"] = decision.feedback
                    self.steering.console.print(f"[yellow]Feedback recorded: {decision.feedback}[/yellow]")
                # Continue to next iteration with feedback

            elif decision.action == SteeringAction.REJECT:
                # Mark task as failed
                next_task.status = "failed"
                next_task.error = decision.feedback or "Rejected by user"
                self.steering.console.print(f"[red]Task rejected: {next_task.description}[/red]")

            elif decision.action == SteeringAction.SKIP:
                # Skip this iteration
                self.steering.console.print("[yellow]Skipping iteration[/yellow]")
                continue

            elif decision.action == SteeringAction.STOP:
                # Stop execution
                break

            # Check completion
            state = self.agent._check_goal_completion(state)

        # Show summary
        summary = self.steering.get_summary()
        self.steering.console.print("\n[bold cyan]═══ Steering Summary ═══[/bold cyan]")
        self.steering.console.print(f"Total decisions: {summary['total_decisions']}")
        self.steering.console.print(f"Actions: {summary['action_counts']}")

        return state
