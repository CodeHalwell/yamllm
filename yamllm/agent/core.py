"""Core Agent class implementing the ReAct loop.

``Agent`` is the original public API and is kept for backwards
compatibility. Since the 2026 modernisation it is a thin wrapper around
:class:`yamllm.agent.harness.AgentHarness`, which owns the ReAct loop and
adds typed events, budgets, cancellation, approvals, and checkpointing.
"""

import logging
from typing import Optional, Dict, Any, Callable

from .models import AgentState, Task, TaskStatus
from .planner import TaskPlanner
from .reasoner import Reasoner
from .actor import Actor
from .observer import Observer
from .recording import SessionRecorder


class Agent:
    """
    Autonomous agent implementing the ReAct loop.

    Coordinates between Planner, Reasoner, Actor, and Observer
    to complete complex tasks autonomously. Execution is delegated to
    :class:`~yamllm.agent.harness.AgentHarness`; use the harness directly
    when you need events, budgets, approvals, or checkpoint/resume.
    """

    def __init__(
        self,
        llm,
        max_iterations: int = 10,
        enable_planning: bool = True,
        enable_reflection: bool = True,
        progress_callback: Optional[Callable[[AgentState], None]] = None,
        logger: Optional[logging.Logger] = None,
        enable_recording: bool = False,
        recording_dir: Optional[str] = None,
        repo_path: Optional[str] = None
    ):
        """
        Initialize the agent.

        Args:
            llm: LLM instance for agent operations
            max_iterations: Maximum number of iterations before stopping
            enable_planning: Whether to use task planning
            enable_reflection: Whether to enable periodic reflection
            progress_callback: Optional callback for progress updates
            logger: Optional logger instance
            enable_recording: Whether to record sessions for replay
            recording_dir: Directory to save recordings (default: ./recordings)
            repo_path: Optional repository path for git operations
        """
        self.llm = llm
        self.max_iterations = max_iterations
        self.enable_planning = enable_planning
        self.enable_reflection = enable_reflection
        self.progress_callback = progress_callback
        self.logger = logger or logging.getLogger(__name__)
        self.enable_recording = enable_recording
        self.recording_dir = recording_dir or "./recordings"
        self.repo_path = repo_path

        # Initialize components
        self.planner = TaskPlanner(llm, logger)
        self.reasoner = Reasoner(llm, logger)
        self.actor = Actor(llm, logger)
        self.observer = Observer(llm, logger)

        # Session recorder (populated by the harness after execute())
        self.recorder: Optional[SessionRecorder] = None

        # Advanced git workflow (P1)
        self.git_workflow = None
        if repo_path:
            try:
                from yamllm.tools.advanced_git import AdvancedGitWorkflow
                self.git_workflow = AdvancedGitWorkflow(repo_path, llm, logger)
            except Exception as e:
                self.logger.warning(f"Could not initialize git workflow: {e}")

    def _build_harness(self):
        """Create a harness sharing this agent's components and settings."""
        from .harness import AgentHarness

        return AgentHarness(
            self.llm,
            max_iterations=self.max_iterations,
            enable_planning=self.enable_planning,
            enable_reflection=self.enable_reflection,
            enable_recording=self.enable_recording,
            recording_dir=self.recording_dir,
            planner=self.planner,
            reasoner=self.reasoner,
            actor=self.actor,
            observer=self.observer,
            logger=self.logger,
        )

    def execute(self, goal: str, context: Optional[Dict[str, Any]] = None) -> AgentState:
        """
        Execute the agentic loop to achieve the given goal.

        Args:
            goal: High-level goal to achieve
            context: Optional context (files, repo info, etc.)

        Returns:
            Final AgentState with results
        """
        self.logger.info(f"Agent starting execution for goal: {goal}")

        harness = self._build_harness()
        if self.progress_callback:
            harness.on_state_change(self.progress_callback)

        state = harness.run(goal, context)
        self.recorder = harness.recorder
        return state

    def _check_goal_completion(self, state: AgentState) -> AgentState:
        """Check if the goal has been completed (kept for legacy callers)."""
        return self._build_harness()._check_goal_completion(state)

    def _reflect(self, state: AgentState) -> AgentState:
        """Reflect on progress (kept for legacy callers)."""
        return self._build_harness()._reflect(state)

    def _notify_progress(self, state: AgentState) -> None:
        """Notify progress callback if set."""
        if self.progress_callback:
            try:
                self.progress_callback(state)
            except Exception as e:
                self.logger.error(f"Progress callback failed: {e}")


class SimpleAgent(Agent):
    """
    Simplified agent for single-task execution without planning.

    Useful for quick tasks that don't need decomposition.
    """

    def __init__(
        self,
        llm,
        max_iterations: int = 3,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize simple agent."""
        super().__init__(
            llm=llm,
            max_iterations=max_iterations,
            enable_planning=False,
            enable_reflection=False,
            logger=logger
        )

    def execute(self, goal: str, context: Optional[Dict[str, Any]] = None) -> AgentState:
        """Execute goal as a single task."""
        self.logger.info(f"SimpleAgent executing: {goal}")

        # Create single task
        state = AgentState(
            goal=goal,
            tasks=[Task.create(goal)],
            max_iterations=self.max_iterations,
            metadata=context or {}
        )

        # Execute task directly
        task = state.tasks[0]
        try:
            action_result = self.actor.act(task, state)
        except Exception as exc:
            self.logger.error(f"SimpleAgent execution failed: {exc}")
            state.completed = True
            state.success = False
            state.error = str(exc)
            self._notify_progress(state)
            return state

        state.add_action(action_result.to_dict())

        # Observe result
        state = self.observer.observe(action_result, state)

        # Set completion
        state.completed = True
        state.success = action_result.success
        if not action_result.success and not state.error:
            state.error = action_result.error or "Task did not succeed"

        self._notify_progress(state)
        return state

    def _notify_progress(self, state: AgentState) -> None:
        if self.progress_callback:
            try:
                self.progress_callback(state)
            except Exception as exc:
                self.logger.warning(f"progress_callback raised: {exc}")


__all__ = ["Agent", "SimpleAgent", "TaskStatus"]
