"""Core Agent class implementing the ReAct loop."""

import logging
from typing import Optional, Dict, Any, Callable
from pathlib import Path

from .models import AgentState, Task, TaskStatus
from .planner import TaskPlanner
from .reasoner import Reasoner
from .actor import Actor
from .observer import Observer
from .recording import SessionRecorder


class Agent:
    """
    Autonomous agent implementing ReAct loop.

    Coordinates between Planner, Reasoner, Actor, and Observer
    to complete complex tasks autonomously.
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

        # Session recorder
        self.recorder: Optional[SessionRecorder] = None

        # Advanced git workflow (P1)
        self.git_workflow = None
        if repo_path:
            try:
                from yamllm.tools.advanced_git import AdvancedGitWorkflow
                self.git_workflow = AdvancedGitWorkflow(repo_path, llm, logger)
            except Exception as e:
                self.logger.warning(f"Could not initialize git workflow: {e}")

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

        # Initialize state
        state = AgentState(
            goal=goal,
            tasks=[],
            current_task_id=None,
            iteration=0,
            max_iterations=self.max_iterations,
            thought_history=[],
            action_history=[],
            completed=False,
            success=False,
            metadata=context or {}
        )

        # Initialize session recorder with agent state
        if self.enable_recording:
            self.recorder = SessionRecorder(state)
            self.logger.info(f"Recording enabled: session {self.recorder.recording['session_id']}")

        try:
            # Phase 1: Planning
            if self.enable_planning:
                self.logger.info("Phase 1: Planning tasks")
                state = self.planner.decompose_goal(goal, context, state)
                self._notify_progress(state)

                if not state.tasks:
                    self.logger.warning("No tasks created, goal may be too vague")
                    state.completed = True
                    state.success = False
                    state.error = "Could not decompose goal into tasks"
                    return state

            # Phase 2: Execution Loop (ReAct)
            self.logger.info("Phase 2: Executing ReAct loop")
            while not state.completed and state.iteration < state.max_iterations:
                state.iteration += 1
                self.logger.info(f"Iteration {state.iteration}/{state.max_iterations}")

                # Step 1: REASON - What should I do next?
                thought, next_task = self.reasoner.reason(state)
                state.add_thought(thought)

                if next_task is None:
                    # No tasks available
                    self.logger.info("No more tasks available")
                    state = self._check_goal_completion(state)
                    break

                state.current_task_id = next_task.id
                self._notify_progress(state)

                # Step 2: ACT - Execute the action
                action_result = self.actor.act(next_task, state)
                state.add_action(action_result.to_dict())
                self._notify_progress(state)

                # Step 3: OBSERVE - Interpret results and update state
                state = self.observer.observe(action_result, state)
                self._notify_progress(state)

                # Record iteration if enabled
                if self.recorder:
                    self.recorder.record_iteration(
                        iteration=state.iteration,
                        thought=thought,
                        action={
                            "task_id": next_task.id,
                            "description": next_task.description,
                            "result": action_result.to_dict()
                        },
                        observation={
                            "completed_tasks": len(state.get_completed_tasks()),
                            "progress": state.get_progress()
                        }
                    )

                # Step 4: Check completion
                state = self._check_goal_completion(state)

                # Optional: REFLECT - Learn from errors/successes
                if self.enable_reflection and state.iteration % 3 == 0:
                    state = self._reflect(state)

            # Check if we hit max iterations
            if state.iteration >= state.max_iterations and not state.completed:
                self.logger.warning("Reached max iterations without completing goal")
                state.completed = True
                state.success = False
                state.error = "Maximum iterations reached"

            self.logger.info(f"Agent execution completed. Success: {state.success}")

            # Save recording if enabled
            if self.recorder:
                try:
                    # Create recording directory if needed
                    Path(self.recording_dir).mkdir(parents=True, exist_ok=True)

                    # Save recording
                    recording_path = Path(self.recording_dir) / f"{self.recorder.session_id}.yaml"
                    self.recorder.save(str(recording_path))
                    self.logger.info(f"Session recording saved to: {recording_path}")

                    # Add recording path to state metadata
                    state.metadata["recording_path"] = str(recording_path)
                except Exception as e:
                    self.logger.error(f"Error saving recording: {e}")

        except Exception as e:
            self.logger.error(f"Agent execution failed: {e}", exc_info=True)
            state.completed = True
            state.success = False
            state.error = str(e)

            # Save recording even on failure
            if self.recorder:
                try:
                    Path(self.recording_dir).mkdir(parents=True, exist_ok=True)
                    recording_path = Path(self.recording_dir) / f"{self.recorder.session_id}_failed.yaml"
                    self.recorder.save(str(recording_path))
                    self.logger.info(f"Failed session recording saved to: {recording_path}")
                except Exception as rec_err:
                    self.logger.error(f"Error saving failed recording: {rec_err}")

        return state

    def _check_goal_completion(self, state: AgentState) -> AgentState:
        """
        Check if the goal has been completed.

        Args:
            state: Current agent state

        Returns:
            Updated state with completion status
        """
        # Count completed and failed tasks
        completed = [t for t in state.tasks if t.status == TaskStatus.COMPLETED]
        failed = [t for t in state.tasks if t.status == TaskStatus.FAILED]
        pending = [t for t in state.tasks if t.status in [TaskStatus.PENDING, TaskStatus.BLOCKED]]

        # Goal is completed if all tasks are either completed or failed
        if not pending:
            state.completed = True

            # Success if at least 50% of tasks completed
            if len(completed) >= len(state.tasks) / 2:
                state.success = True
                self.logger.info(f"Goal achieved: {len(completed)}/{len(state.tasks)} tasks completed")
            else:
                state.success = False
                state.error = f"Too many failed tasks: {len(failed)}/{len(state.tasks)}"
                self.logger.warning(state.error)

        return state

    def _reflect(self, state: AgentState) -> AgentState:
        """
        Reflect on progress and adjust strategy if needed.

        Args:
            state: Current agent state

        Returns:
            Updated state
        """
        self.logger.info("Reflecting on progress...")

        # Simple reflection: check if we're making progress
        completed_count = len(state.get_completed_tasks())
        failed_count = len([t for t in state.tasks if t.status == TaskStatus.FAILED])

        if failed_count > completed_count:
            thought = "Too many failures. May need to adjust approach or seek help."
            state.add_thought(thought)
            self.logger.warning(thought)

        elif completed_count == 0 and state.iteration > 2:
            thought = "No progress made. May need to simplify tasks or change strategy."
            state.add_thought(thought)
            self.logger.warning(thought)

        else:
            thought = f"Making good progress: {completed_count} tasks completed."
            state.add_thought(thought)
            self.logger.info(thought)

        return state

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
        action_result = self.actor.act(task, state)
        state.add_action(action_result.to_dict())

        # Observe result
        state = self.observer.observe(action_result, state)

        # Set completion
        state.completed = True
        state.success = action_result.success

        return state
