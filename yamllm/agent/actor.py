"""Action component for executing tasks."""

import logging
import time
from typing import Dict, Any, Optional

from .models import Task, AgentState, TaskStatus, ActionResult


class Actor:
    """
    Action component - executes tasks.

    Implements the 'Act' part of ReAct.
    """

    def __init__(self, llm, logger: Optional[logging.Logger] = None):
        """
        Initialize the actor.

        Args:
            llm: LLM instance for execution
            logger: Optional logger instance
        """
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)

    def act(self, task: Task, state: AgentState) -> ActionResult:
        """
        Execute a task.

        Args:
            task: Task to execute
            state: Current agent state

        Returns:
            ActionResult with execution details
        """
        self.logger.info(f"Executing task {task.id}: {task.description}")

        # Update task status
        task.status = TaskStatus.IN_PROGRESS

        # Build action prompt
        prompt = self._build_action_prompt(task, state)

        start_time = time.time()

        try:
            # Execute with tools enabled
            response = self.llm.get_completion_with_tools(prompt)

            execution_time = time.time() - start_time

            # Extract result
            result = ActionResult(
                task_id=task.id,
                success=True,
                response=response.get("content", ""),
                tool_calls=response.get("tool_calls", []),
                tool_results=response.get("tool_results", []),
                error=None,
                execution_time=execution_time
            )

            # Update task
            task.status = TaskStatus.COMPLETED
            task.result = result.to_dict()

            self.logger.info(f"Task {task.id} completed in {execution_time:.2f}s")

        except Exception as e:
            execution_time = time.time() - start_time

            self.logger.error(f"Task {task.id} failed: {e}")

            result = ActionResult(
                task_id=task.id,
                success=False,
                response=None,
                tool_calls=[],
                tool_results=[],
                error=str(e),
                execution_time=execution_time
            )

            task.status = TaskStatus.FAILED
            task.error = str(e)

        return result

    def _build_action_prompt(self, task: Task, state: AgentState) -> str:
        """Build prompt for action execution."""
        # Get context from completed tasks
        completed_context = self._format_completed_tasks(state)

        # Get suggested tools if available
        suggested_tools = task.metadata.get("tools", [])
        tools_hint = f"\nSuggested tools: {', '.join(suggested_tools)}" if suggested_tools else ""

        return f"""You are working on achieving this goal: {state.goal}

Current Task: {task.description}

Context from previous completed tasks:
{completed_context}
{tools_hint}

Execute this task using the available tools. Be specific, thorough, and focus on completing just this one task.

Important:
- Use tools to gather information or perform actions
- Provide clear, actionable output
- If the task cannot be completed, explain why"""

    def _format_completed_tasks(self, state: AgentState) -> str:
        """Format completed tasks for context."""
        completed = state.get_completed_tasks()

        if not completed:
            return "No tasks completed yet."

        lines = []
        for task in completed:
            result = task.result or {}
            response = result.get("response", "No response")
            # Truncate long responses
            if len(response) > 200:
                response = response[:200] + "..."
            lines.append(f"- {task.id}: {task.description}\n  Result: {response}")

        return "\n".join(lines)

    def _list_available_tools(self) -> str:
        """List available tools (if LLM has tool manager)."""
        try:
            if hasattr(self.llm, 'tool_orchestrator') and self.llm.tool_orchestrator:
                tools = self.llm.tool_orchestrator.tool_manager.list()
                return ", ".join(tools)
        except Exception as e:
            self.logger.warning(f"Failed to list available tools: {e}")
        return "Tools available through LLM"
