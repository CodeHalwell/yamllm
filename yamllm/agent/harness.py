"""Modern agent harness: an event-driven ReAct engine.

This is the 2026-era replacement for the original ``Agent.execute`` loop.
The engine itself never touches a console. Instead it:

- emits typed :class:`~yamllm.agent.events.AgentEvent` objects to any number
  of subscribers (Textual TUI, plain CLI renderer, JSONL log sinks),
- enforces budgets (iterations, wall-clock time, consecutive failures),
- supports cooperative cancellation from another thread,
- gates actions behind a pluggable human-in-the-loop approval policy,
- checkpoints state after every iteration so runs can be resumed.

The blocking ``run()`` method is designed to execute on a worker thread while
a UI event loop (e.g. Textual's asyncio loop) renders the event stream.
"""

import logging
import json
import threading
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .actor import Actor
from .events import AgentEvent, EventKind
from .interactive_steering import SteeringAction, SteeringDecision, SteeringPoint
from .models import AgentState, Task, TaskStatus
from .observer import Observer
from .planner import TaskPlanner
from .reasoner import Reasoner
from .recording import SessionRecorder

EventListener = Callable[[AgentEvent], None]
StateListener = Callable[[AgentState], None]
DecisionProvider = Callable[[SteeringPoint], SteeringDecision]
Watchpoint = Callable[[SteeringPoint], bool]


class ApprovalPolicy(str, Enum):
    """When the harness should pause for a human decision."""

    NEVER = "never"
    ALWAYS = "always"
    ON_WATCHPOINT = "on_watchpoint"


def load_checkpoint(path: str) -> AgentState:
    """Load an :class:`AgentState` from a checkpoint file."""
    with open(path, "r") as f:
        return AgentState.from_dict(json.load(f))


class AgentHarness:
    """Event-driven ReAct engine coordinating planner, reasoner, actor, observer."""

    def __init__(
        self,
        llm: Any,
        *,
        max_iterations: int = 10,
        max_wall_time: Optional[float] = None,
        max_consecutive_failures: Optional[int] = None,
        enable_planning: bool = True,
        enable_reflection: bool = True,
        success_threshold: float = 0.5,
        approval_policy: ApprovalPolicy = ApprovalPolicy.NEVER,
        decision_provider: Optional[DecisionProvider] = None,
        checkpoint_dir: Optional[str] = None,
        enable_recording: bool = False,
        recording_dir: Optional[str] = None,
        planner: Optional[TaskPlanner] = None,
        reasoner: Optional[Reasoner] = None,
        actor: Optional[Actor] = None,
        observer: Optional[Observer] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the harness.

        Args:
            llm: LLM instance used by all components.
            max_iterations: Iteration budget for the ReAct loop.
            max_wall_time: Optional wall-clock budget in seconds.
            max_consecutive_failures: Stop after this many failed actions in a row.
            enable_planning: Decompose the goal into tasks before executing.
            enable_reflection: Periodically reflect on progress.
            success_threshold: Fraction of tasks that must complete for success.
            approval_policy: When to pause for a human decision.
            decision_provider: Blocking callable that returns a human decision.
                Required for ALWAYS/ON_WATCHPOINT policies; without one the
                harness auto-approves and logs a warning.
            checkpoint_dir: Directory for per-iteration state checkpoints.
            enable_recording: Record the session for later replay.
            recording_dir: Directory for session recordings.
            planner/reasoner/actor/observer: Optional pre-built components
                (shared with a legacy ``Agent`` wrapper, or customised).
            logger: Optional logger.
        """
        self.llm = llm
        self.max_iterations = max_iterations
        self.max_wall_time = max_wall_time
        self.max_consecutive_failures = max_consecutive_failures
        self.enable_planning = enable_planning
        self.enable_reflection = enable_reflection
        self.success_threshold = success_threshold
        self.approval_policy = approval_policy
        self.decision_provider = decision_provider
        self.checkpoint_dir = checkpoint_dir
        self.enable_recording = enable_recording
        self.recording_dir = recording_dir or "./recordings"
        self.logger = logger or logging.getLogger(__name__)

        self.planner = planner or TaskPlanner(llm, self.logger)
        self.reasoner = reasoner or Reasoner(llm, self.logger)
        self.actor = actor or Actor(llm, self.logger)
        self.observer = observer or Observer(llm, self.logger)

        self.recorder: Optional[SessionRecorder] = None
        self.run_id: Optional[str] = None

        self._listeners: List[EventListener] = []
        self._state_listeners: List[StateListener] = []
        self._watchpoints: List[Watchpoint] = []
        self._stop_event = threading.Event()
        self._auto_approve = False
        self.decision_history: List[SteeringDecision] = []

    # ------------------------------------------------------------------
    # Subscription / control surface
    # ------------------------------------------------------------------
    def add_listener(self, listener: EventListener) -> None:
        """Subscribe to the run's event stream."""
        self._listeners.append(listener)

    def on_state_change(self, listener: StateListener) -> None:
        """Subscribe to coarse state snapshots (legacy progress callbacks)."""
        self._state_listeners.append(listener)

    def add_watchpoint(self, condition: Watchpoint) -> None:
        """Pause for approval when ``condition`` matches a steering point."""
        self._watchpoints.append(condition)

    def request_stop(self) -> None:
        """Cooperatively cancel the run (safe to call from another thread)."""
        self._stop_event.set()

    @property
    def stop_requested(self) -> bool:
        """Whether cancellation has been requested."""
        return self._stop_event.is_set()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        initial_state: Optional[AgentState] = None,
    ) -> AgentState:
        """
        Run the ReAct loop until the goal completes or a budget is exhausted.

        Args:
            goal: High-level goal to achieve.
            context: Optional context (files, repo info, etc.).
            initial_state: Resume from a previously checkpointed state
                instead of planning from scratch.

        Returns:
            Final :class:`AgentState`.
        """
        self._stop_event.clear()
        self._auto_approve = False
        self.run_id = uuid.uuid4().hex[:8]
        started_at = time.monotonic()

        if initial_state is not None:
            state = initial_state
            # The configured limit is fresh capacity for the resumed run, so
            # an iteration-exhausted checkpoint can actually continue.
            state.max_iterations = max(
                state.max_iterations, state.iteration + self.max_iterations
            )
            if context:
                # Freshly supplied context overrides stale checkpoint metadata
                state.metadata.update(context)
            if state.completed and not state.success and state.get_pending_tasks():
                # A stopped or budget-exhausted checkpoint resumes as a live
                # run rather than immediately returning its terminal state.
                state.completed = False
                state.error = None
        else:
            state = AgentState(
                goal=goal,
                tasks=[],
                max_iterations=self.max_iterations,
                metadata=context or {},
            )

        if self.enable_recording:
            self.recorder = SessionRecorder(state)
            self.logger.info(
                f"Recording enabled: session {self.recorder.recording['session_id']}"
            )

        self._emit(
            EventKind.RUN_STARTED,
            {
                "run_id": self.run_id,
                "goal": state.goal,
                "max_iterations": state.max_iterations,
                "approval_policy": self.approval_policy.value,
                "resumed": initial_state is not None,
            },
        )

        try:
            if self.enable_planning and not state.tasks:
                state = self.planner.decompose_goal(state.goal, context, state)
                self._emit(
                    EventKind.PLAN_CREATED,
                    {"tasks": [t.to_dict() for t in state.tasks]},
                )
                self._notify_state(state)

                if not state.tasks:
                    state.completed = True
                    state.success = False
                    state.error = "Could not decompose goal into tasks"
                    self._finish(state)
                    return state
            elif state.tasks:
                self._emit(
                    EventKind.PLAN_CREATED,
                    {"tasks": [t.to_dict() for t in state.tasks]},
                )

            while not state.completed and state.iteration < state.max_iterations:
                if self._stop_event.is_set():
                    state.completed = True
                    state.success = False
                    state.error = "Stopped by user"
                    self._save_checkpoint(state)
                    break

                budget_error = self._check_budgets(state, started_at)
                if budget_error:
                    state.completed = True
                    state.success = False
                    state.error = budget_error
                    break

                state.iteration += 1
                self._emit(
                    EventKind.ITERATION_STARTED,
                    {
                        "iteration": state.iteration,
                        "max_iterations": state.max_iterations,
                    },
                )

                # REASON
                thought, next_task = self.reasoner.reason(state)
                state.add_thought(thought)
                self._emit(EventKind.THOUGHT, {"thought": thought})

                if next_task is None:
                    self._fail_unreachable_tasks(state)
                    state = self._check_goal_completion(state)
                    if not state.completed:
                        # Defensive: nothing runnable but completion didn't
                        # trigger (e.g. residual dependency cycle).
                        state.completed = True
                        state.success = False
                        state.error = "No runnable tasks remain"
                    break

                state.current_task_id = next_task.id
                self._notify_state(state)

                # Optional human-in-the-loop gate
                decision = self._maybe_request_decision(thought, next_task, state)
                if decision is not None:
                    outcome = self._apply_decision(decision, next_task, state)
                    if outcome == "stop":
                        state.completed = True
                        state.success = False
                        state.error = state.error or "Stopped by user"
                        self._save_checkpoint(state)
                        break
                    if outcome == "skip":
                        state = self._check_goal_completion(state)
                        self._save_checkpoint(state)
                        continue

                # ACT
                self._emit(EventKind.ACTION_STARTED, {"task": next_task.to_dict()})
                action_result = self.actor.act(next_task, state)
                # Operator guidance from a MODIFY decision applies to the
                # action it modified, not every later one.
                state.metadata.pop("user_feedback", None)
                state.add_action(action_result.to_dict())
                self._emit(
                    EventKind.ACTION_FINISHED,
                    {"task": next_task.to_dict(), "result": action_result.to_dict()},
                )
                self._emit_task_update(next_task, state)
                self._notify_state(state)

                # OBSERVE
                state = self.observer.observe(action_result, state)
                observations = state.metadata.get("observations") or []
                if observations:
                    self._emit(EventKind.OBSERVATION, dict(observations[-1]))
                self._notify_state(state)

                if self.recorder:
                    self.recorder.record_iteration(
                        iteration=state.iteration,
                        thought=thought,
                        action={
                            "task_id": next_task.id,
                            "description": next_task.description,
                            "result": action_result.to_dict(),
                        },
                        observation={
                            "completed_tasks": len(state.get_completed_tasks()),
                            "progress": state.get_progress(),
                        },
                    )

                state = self._check_goal_completion(state)
                self._save_checkpoint(state)

                if self.enable_reflection and state.iteration % 3 == 0:
                    state = self._reflect(state)

            if state.iteration >= state.max_iterations and not state.completed:
                state.completed = True
                state.success = False
                state.error = "Maximum iterations reached"

            self._finish(state)

        except Exception as e:
            self.logger.error(f"Agent execution failed: {e}", exc_info=True)
            state.completed = True
            state.success = False
            state.error = str(e)
            self._emit(EventKind.ERROR, {"error": str(e)})
            self._finish(state, failed=True)

        return state

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _check_budgets(self, state: AgentState, started_at: float) -> Optional[str]:
        """Return an error message when a budget is exhausted, else None."""
        if self.max_wall_time is not None:
            elapsed = time.monotonic() - started_at
            if elapsed > self.max_wall_time:
                message = f"Wall-time budget exceeded ({elapsed:.0f}s > {self.max_wall_time:.0f}s)"
                self._emit(
                    EventKind.BUDGET_EXCEEDED,
                    {"budget": "wall_time", "detail": message},
                )
                return message

        if self.max_consecutive_failures is not None:
            failures = 0
            for action in reversed(state.action_history):
                if action.get("success"):
                    break
                failures += 1
            if failures >= self.max_consecutive_failures:
                message = f"{failures} consecutive action failures"
                self._emit(
                    EventKind.BUDGET_EXCEEDED,
                    {"budget": "consecutive_failures", "detail": message},
                )
                return message

        return None

    def _maybe_request_decision(
        self, thought: str, task: Task, state: AgentState
    ) -> Optional[SteeringDecision]:
        """Ask the decision provider for approval when the policy requires it."""
        if self.approval_policy == ApprovalPolicy.NEVER or self._auto_approve:
            return None

        # A pending stop should unwind the loop, not enter a (possibly
        # blocking) approval flow.
        if self._stop_event.is_set():
            return SteeringDecision(action=SteeringAction.STOP)

        point = SteeringPoint(
            iteration=state.iteration,
            thought=thought,
            planned_action={
                "task_id": task.id,
                "description": task.description,
                "dependencies": task.dependencies,
            },
            current_state=state,
            context={
                "completed_tasks": len(state.get_completed_tasks()),
                "progress": f"{state.get_progress():.0f}%",
            },
        )

        triggered = any(self._safe_watchpoint(wp, point) for wp in self._watchpoints)
        if self.approval_policy == ApprovalPolicy.ON_WATCHPOINT and not triggered:
            return None

        if self.decision_provider is None:
            self.logger.warning(
                "Approval required but no decision provider configured; auto-approving"
            )
            return None

        self._emit(
            EventKind.APPROVAL_REQUESTED,
            {
                "iteration": point.iteration,
                "thought": point.thought,
                "planned_action": point.planned_action,
                "watchpoint_triggered": triggered,
            },
        )
        decision = self.decision_provider(point)
        self.decision_history.append(decision)
        self._emit(
            EventKind.DECISION,
            {"action": decision.action.value, "feedback": decision.feedback},
        )
        return decision

    def _safe_watchpoint(self, watchpoint: Watchpoint, point: SteeringPoint) -> bool:
        try:
            return bool(watchpoint(point))
        except Exception as e:
            self.logger.warning(f"Watchpoint check failed: {e}")
            return False

    def _apply_decision(
        self, decision: SteeringDecision, task: Task, state: AgentState
    ) -> str:
        """Apply a human decision. Returns 'proceed', 'skip', or 'stop'."""
        if decision.action == SteeringAction.APPROVE:
            return "proceed"

        if decision.action == SteeringAction.AUTO:
            self._auto_approve = True
            return "proceed"

        if decision.action == SteeringAction.MODIFY:
            if decision.feedback:
                state.metadata["user_feedback"] = decision.feedback
            if decision.modified_task:
                task.description = decision.modified_task
            return "proceed"

        if decision.action == SteeringAction.REJECT:
            task.status = TaskStatus.FAILED
            task.error = decision.feedback or "Rejected by user"
            self._emit_task_update(task, state)
            return "skip"

        if decision.action == SteeringAction.SKIP:
            return "skip"

        if decision.action == SteeringAction.STOP:
            return "stop"

        return "proceed"

    def _fail_unreachable_tasks(self, state: AgentState) -> None:
        """Mark tasks terminal when a (transitive) dependency has failed.

        Without this, a failed dependency leaves its dependents pending
        forever and the run would end neither completed nor errored.
        """
        changed = True
        while changed:
            changed = False
            failed_ids = {t.id for t in state.tasks if t.status == TaskStatus.FAILED}
            for task in state.tasks:
                if task.status not in (TaskStatus.PENDING, TaskStatus.BLOCKED):
                    continue
                blocking = [d for d in task.dependencies if d in failed_ids]
                if blocking:
                    task.status = TaskStatus.FAILED
                    task.error = f"Blocked by failed dependency: {', '.join(blocking)}"
                    self._emit_task_update(task, state)
                    changed = True

    def _emit_task_update(self, task: Task, state: AgentState) -> None:
        self._emit(
            EventKind.TASK_UPDATED,
            {"task": task.to_dict(), "progress": state.get_progress()},
        )

    def _save_checkpoint(self, state: AgentState) -> None:
        if not self.checkpoint_dir:
            return
        try:
            directory = Path(self.checkpoint_dir)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{self.run_id}.json"
            with open(path, "w") as f:
                json.dump(state.to_dict(), f, indent=2, default=str)
            state.metadata["checkpoint_path"] = str(path)
            self._emit(EventKind.CHECKPOINT_SAVED, {"path": str(path)})
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint: {e}")

    def _check_goal_completion(self, state: AgentState) -> AgentState:
        """Mark the run complete when no runnable tasks remain."""
        completed = [t for t in state.tasks if t.status == TaskStatus.COMPLETED]
        failed = [t for t in state.tasks if t.status == TaskStatus.FAILED]
        pending = [
            t
            for t in state.tasks
            if t.status in (TaskStatus.PENDING, TaskStatus.BLOCKED)
        ]

        if not pending:
            state.completed = True
            if (
                state.tasks
                and len(completed) >= len(state.tasks) * self.success_threshold
            ):
                state.success = True
                self.logger.info(
                    f"Goal achieved: {len(completed)}/{len(state.tasks)} tasks completed"
                )
            else:
                state.success = False
                state.error = f"Too many failed tasks: {len(failed)}/{len(state.tasks)}"
                self.logger.warning(state.error)

        return state

    def _reflect(self, state: AgentState) -> AgentState:
        """Lightweight self-assessment recorded into the thought history."""
        completed_count = len(state.get_completed_tasks())
        failed_count = len([t for t in state.tasks if t.status == TaskStatus.FAILED])

        if failed_count > completed_count:
            thought = "Too many failures. May need to adjust approach or seek help."
        elif completed_count == 0 and state.iteration > 2:
            thought = "No progress made. May need to simplify tasks or change strategy."
        else:
            thought = f"Making good progress: {completed_count} tasks completed."

        state.add_thought(thought)
        self._emit(EventKind.THOUGHT, {"thought": thought, "reflection": True})
        return state

    def _finish(self, state: AgentState, failed: bool = False) -> None:
        """Emit the final event and persist the session recording."""
        self._emit(
            EventKind.RUN_FINISHED,
            {
                "success": state.success,
                "completed": state.completed,
                "iterations": state.iteration,
                "error": state.error,
                "tasks_completed": len(state.get_completed_tasks()),
                "tasks_total": len(state.tasks),
            },
        )
        self._notify_state(state)

        if not self.recorder:
            return
        try:
            Path(self.recording_dir).mkdir(parents=True, exist_ok=True)
            suffix = "_failed" if failed else ""
            recording_path = (
                Path(self.recording_dir) / f"{self.recorder.session_id}{suffix}.yaml"
            )
            self.recorder.finalize(success=state.success, error=state.error)
            self.recorder.save(str(recording_path))
            state.metadata["recording_path"] = str(recording_path)
            self.logger.info(f"Session recording saved to: {recording_path}")
        except Exception as e:
            self.logger.error(f"Error saving recording: {e}")

    def _emit(self, kind: EventKind, payload: Dict[str, Any]) -> None:
        event = AgentEvent(kind=kind, payload=payload)
        for listener in self._listeners:
            try:
                listener(event)
            except Exception as e:
                self.logger.error(f"Event listener failed: {e}")

    def _notify_state(self, state: AgentState) -> None:
        for listener in self._state_listeners:
            try:
                listener(state)
            except Exception as e:
                self.logger.error(f"State listener failed: {e}")
