"""Textual TUI for live agent runs.

Replaces the old Rich ``Prompt.ask`` steering loop with a full-screen,
event-driven interface: a live task board, a streaming transcript of
thoughts/actions/observations, and a modal approval dialog for
human-in-the-loop control.

The :class:`~yamllm.agent.harness.AgentHarness` runs on a worker thread and
publishes :class:`~yamllm.agent.events.AgentEvent` objects; the app renders
them on the UI thread via ``call_from_thread``. Approval requests block the
worker on a queue until the operator decides in the modal.
"""

from __future__ import annotations

import queue
from typing import Any, Dict, Optional

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Header, Input, Label, RichLog, Static

from yamllm.agent.events import AgentEvent, EventKind
from yamllm.agent.harness import AgentHarness, ApprovalPolicy
from yamllm.agent.interactive_steering import (
    SteeringAction,
    SteeringDecision,
    SteeringPoint,
)
from yamllm.agent.models import AgentState

STATUS_ICONS = {
    "pending": "⋯",
    "in_progress": "▶",
    "completed": "✓",
    "failed": "✗",
    "blocked": "⊗",
}

STATUS_STYLES = {
    "pending": "dim",
    "in_progress": "yellow",
    "completed": "green",
    "failed": "red",
    "blocked": "dim",
}


class ApprovalScreen(ModalScreen[SteeringDecision]):
    """Modal dialog asking the operator to approve the agent's next action."""

    # Keep focus off the feedback Input so single-key decisions work
    # immediately; 'm' (or clicking) focuses it, Escape leaves it.
    # ("" disables auto-focus; None would inherit the app default.)
    AUTO_FOCUS = ""

    BINDINGS = [
        Binding("a", "approve", "Approve"),
        Binding("r", "reject", "Reject"),
        Binding("s", "skip", "Skip"),
        Binding("m", "modify", "Modify"),
        Binding("o", "auto", "Auto-approve rest"),
        Binding("x", "stop", "Stop run"),
    ]

    def __init__(self, point: SteeringPoint) -> None:
        super().__init__()
        self.point = point

    def compose(self) -> ComposeResult:
        action = self.point.planned_action or {}
        with Vertical(id="approval-dialog"):
            yield Label(
                f"Iteration {self.point.iteration} — approval required",
                id="approval-title",
            )
            yield Static(
                Text(self.point.thought or "(no reasoning provided)", style="italic"),
                id="approval-thought",
            )
            yield Static(
                Text.assemble(
                    ("Next action: ", "bold"),
                    (str(action.get("description", "?")), ""),
                    ("\nTask id: ", "bold"),
                    (str(action.get("task_id", "?")), "dim"),
                ),
                id="approval-action",
            )
            yield Input(
                placeholder="Guidance for the agent (press Enter to send as 'modify')",
                id="approval-feedback",
            )
            yield Label(
                "a approve · r reject · s skip · m modify · o auto · x stop",
                id="approval-keys",
            )

    def action_approve(self) -> None:
        self.dismiss(SteeringDecision(action=SteeringAction.APPROVE))

    def action_reject(self) -> None:
        feedback = self.query_one("#approval-feedback", Input).value or None
        self.dismiss(SteeringDecision(action=SteeringAction.REJECT, feedback=feedback))

    def action_skip(self) -> None:
        self.dismiss(SteeringDecision(action=SteeringAction.SKIP))

    def action_auto(self) -> None:
        self.dismiss(SteeringDecision(action=SteeringAction.AUTO))

    def action_stop(self) -> None:
        self.dismiss(SteeringDecision(action=SteeringAction.STOP))

    def action_modify(self) -> None:
        self.query_one("#approval-feedback", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        feedback = event.value.strip()
        if feedback:
            self.dismiss(
                SteeringDecision(action=SteeringAction.MODIFY, feedback=feedback)
            )

    def on_key(self, event) -> None:
        if event.key == "escape":
            self.set_focus(None)
            event.stop()


class AgentTUI(App[Optional[AgentState]]):
    """Full-screen dashboard for a single agent run."""

    TITLE = "yamllm agent"

    CSS = """
    #body {
        height: 1fr;
    }
    #left {
        width: 42%;
        min-width: 30;
        border: round $primary;
    }
    #goal {
        padding: 0 1;
        height: auto;
        max-height: 6;
        border-bottom: solid $primary;
    }
    #tasks {
        height: 1fr;
    }
    #transcript {
        width: 1fr;
        border: round $secondary;
        padding: 0 1;
    }
    #status {
        dock: bottom;
        height: 1;
        padding: 0 1;
        background: $panel;
        color: $text;
    }
    ApprovalScreen {
        align: center middle;
    }
    #approval-dialog {
        width: 80;
        max-width: 90%;
        height: auto;
        padding: 1 2;
        border: thick $warning;
        background: $surface;
    }
    #approval-title {
        text-style: bold;
        color: $warning;
        margin-bottom: 1;
    }
    #approval-thought {
        margin-bottom: 1;
    }
    #approval-action {
        margin-bottom: 1;
    }
    #approval-keys {
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("x", "stop_run", "Stop run"),
        Binding("q", "request_quit", "Quit"),
    ]

    def __init__(
        self,
        harness: AgentHarness,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        initial_state: Optional[AgentState] = None,
    ) -> None:
        super().__init__()
        self.harness = harness
        self.goal = goal
        self.context = context
        self.initial_state = initial_state
        self.final_state: Optional[AgentState] = None
        self._decision_queue: "queue.Queue[SteeringDecision]" = queue.Queue()
        self._awaiting_decision = False
        self._quit_requested = False
        self._task_rows: Dict[str, Any] = {}
        self._column_keys: list = []

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield Static(Text(self.goal, style="bold"), id="goal")
                yield DataTable(id="tasks", cursor_type="row", zebra_stripes=True)
            yield RichLog(id="transcript", wrap=True, markup=False, highlight=False)
        yield Static("starting…", id="status")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#tasks", DataTable)
        self._column_keys = table.add_columns(" ", "id", "task")

        # Install the modal approval flow only when the caller has not
        # supplied their own provider (e.g. CLI --auto-approve).
        if (
            self.harness.approval_policy != ApprovalPolicy.NEVER
            and self.harness.decision_provider is None
        ):
            self.harness.decision_provider = self._blocking_decision_provider
        self.harness.add_listener(self._event_from_worker)
        self.run_worker(self._run_agent, thread=True, exclusive=True)

    # ------------------------------------------------------------------
    # Worker side (harness thread)
    # ------------------------------------------------------------------
    def _run_agent(self) -> None:
        state = self.harness.run(
            self.goal, self.context, initial_state=self.initial_state
        )
        self.call_from_thread(self._on_run_complete, state)

    def _event_from_worker(self, event: AgentEvent) -> None:
        self.call_from_thread(self._handle_event, event)

    def _blocking_decision_provider(self, point: SteeringPoint) -> SteeringDecision:
        """Called on the worker thread; blocks until the operator decides."""
        self._awaiting_decision = True
        self.call_from_thread(self._open_approval, point)
        decision = self._decision_queue.get()
        self._awaiting_decision = False
        return decision

    # ------------------------------------------------------------------
    # UI side
    # ------------------------------------------------------------------
    def _open_approval(self, point: SteeringPoint) -> None:
        def deliver(decision: Optional[SteeringDecision]) -> None:
            # Fail closed: a dismissal without an explicit decision must not
            # approve the action.
            self._decision_queue.put(
                decision or SteeringDecision(action=SteeringAction.STOP)
            )

        self.push_screen(ApprovalScreen(point), deliver)

    def _handle_event(self, event: AgentEvent) -> None:
        log = self.query_one("#transcript", RichLog)
        payload = event.payload

        if event.kind == EventKind.RUN_STARTED:
            self._set_status("running")
            log.write(Text(f"run {payload.get('run_id', '')} started", style="dim"))

        elif event.kind == EventKind.PLAN_CREATED:
            for task in payload.get("tasks", []):
                self._upsert_task_row(task)
            log.write(
                Text(
                    f"plan created: {len(payload.get('tasks', []))} tasks", style="cyan"
                )
            )

        elif event.kind == EventKind.ITERATION_STARTED:
            self._set_status(
                f"iteration {payload.get('iteration')}/{payload.get('max_iterations')}"
            )

        elif event.kind == EventKind.THOUGHT:
            style = "magenta" if payload.get("reflection") else "italic dim"
            log.write(Text(f"💭 {payload.get('thought', '')}", style=style))

        elif event.kind == EventKind.ACTION_STARTED:
            task = payload.get("task", {})
            self._upsert_task_row(task)
            log.write(Text(f"⚡ {task.get('description', '')}", style="yellow"))

        elif event.kind == EventKind.ACTION_FINISHED:
            task = payload.get("task", {})
            result = payload.get("result", {})
            self._upsert_task_row(task)
            if result.get("success"):
                response = (result.get("response") or "").strip()
                if len(response) > 400:
                    response = response[:400] + "…"
                tool_calls = result.get("tool_calls") or []
                if tool_calls:
                    names = ", ".join(str(c.get("name", "?")) for c in tool_calls[:6])
                    log.write(Text(f"🔧 tools: {names}", style="blue"))
                if response:
                    log.write(Text(response))
            else:
                log.write(Text(f"✗ {result.get('error', 'failed')}", style="red"))

        elif event.kind == EventKind.TASK_UPDATED:
            self._upsert_task_row(payload.get("task", {}))
            progress = payload.get("progress")
            if progress is not None:
                self._set_status(f"progress {progress:.0f}%")

        elif event.kind == EventKind.OBSERVATION:
            learned = payload.get("learned")
            if learned:
                log.write(Text(f"👁 {learned}", style="green"))

        elif event.kind == EventKind.APPROVAL_REQUESTED:
            log.write(Text("⏸ waiting for approval…", style="bold yellow"))

        elif event.kind == EventKind.DECISION:
            log.write(
                Text(
                    f"→ decision: {payload.get('action')}"
                    + (f" ({payload['feedback']})" if payload.get("feedback") else ""),
                    style="yellow",
                )
            )

        elif event.kind == EventKind.BUDGET_EXCEEDED:
            log.write(
                Text(f"⛔ {payload.get('detail', 'budget exceeded')}", style="bold red")
            )

        elif event.kind == EventKind.CHECKPOINT_SAVED:
            log.write(Text(f"💾 checkpoint: {payload.get('path')}", style="dim"))

        elif event.kind == EventKind.ERROR:
            log.write(Text(f"error: {payload.get('error')}", style="bold red"))

        elif event.kind == EventKind.RUN_FINISHED:
            outcome = "✅ success" if payload.get("success") else "❌ not achieved"
            log.write(
                Text(
                    f"{outcome} — {payload.get('tasks_completed')}/"
                    f"{payload.get('tasks_total')} tasks in "
                    f"{payload.get('iterations')} iterations",
                    style="bold",
                )
            )
            if payload.get("error"):
                log.write(Text(str(payload["error"]), style="red"))

    def _upsert_task_row(self, task: Dict[str, Any]) -> None:
        if not task or "id" not in task:
            return
        table = self.query_one("#tasks", DataTable)
        status = task.get("status", "pending")
        icon = Text(STATUS_ICONS.get(status, "?"), style=STATUS_STYLES.get(status, ""))
        description = task.get("description", "")
        if task["id"] in self._task_rows:
            row_key = self._task_rows[task["id"]]
            table.update_cell(row_key, self._column_keys[0], icon)
            table.update_cell(row_key, self._column_keys[2], description)
        else:
            row_key = table.add_row(icon, task["id"], description, key=task["id"])
            self._task_rows[task["id"]] = row_key

    def _set_status(self, message: str) -> None:
        suffix = " · press q to quit" if self.final_state is not None else ""
        self.query_one("#status", Static).update(f"{message}{suffix}")

    def _on_run_complete(self, state: AgentState) -> None:
        self.final_state = state
        self._set_status("finished — success" if state.success else "finished — failed")
        if self._quit_requested:
            self.exit(state)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def action_stop_run(self) -> None:
        if self.final_state is None:
            self.harness.request_stop()
            self._set_status("stopping after current step…")
            if self._awaiting_decision:
                self._decision_queue.put(SteeringDecision(action=SteeringAction.STOP))

    def action_request_quit(self) -> None:
        if self.final_state is not None:
            self.exit(self.final_state)
            return
        # Still running: request a stop and exit once the worker unwinds.
        self._quit_requested = True
        self.harness.request_stop()
        self._set_status("stopping — exiting when the current step finishes…")
        if self._awaiting_decision:
            self._decision_queue.put(SteeringDecision(action=SteeringAction.STOP))


def run_agent_tui(
    harness: AgentHarness,
    goal: str,
    context: Optional[Dict[str, Any]] = None,
    initial_state: Optional[AgentState] = None,
) -> Optional[AgentState]:
    """Run an agent inside the Textual TUI and return the final state."""
    app = AgentTUI(harness, goal, context, initial_state)
    result = app.run()
    return result if result is not None else app.final_state
