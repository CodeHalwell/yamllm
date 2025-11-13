"""Reasoning component for deciding next actions."""

import json
import logging
from typing import Tuple, List, Optional

from .models import Task, AgentState, TaskStatus


class Reasoner:
    """
    Reasoning component - decides next actions.

    Implements the 'Reason' part of ReAct.
    """

    def __init__(self, llm, logger: Optional[logging.Logger] = None):
        """
        Initialize the reasoner.

        Args:
            llm: LLM instance for reasoning
            logger: Optional logger instance
        """
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)

    def reason(self, state: AgentState) -> Tuple[str, Optional[Task]]:
        """
        Reason about next action to take.

        Args:
            state: Current agent state

        Returns:
            (thought, next_task): Reasoning and selected task
        """
        # Get available tasks (not blocked by dependencies)
        available_tasks = state.get_available_tasks()

        if not available_tasks:
            # All tasks blocked or completed
            thought = "No available tasks. All tasks are either completed or blocked by dependencies."
            self.logger.info(thought)
            return thought, None

        # If only one task, select it without reasoning
        if len(available_tasks) == 1:
            task = available_tasks[0]
            thought = f"Only one available task: {task.description}"
            self.logger.info(f"Selected task {task.id}: {task.description}")
            return thought, task

        # Build reasoning prompt
        prompt = self._build_reasoning_prompt(state, available_tasks)

        try:
            # Get LLM's reasoning
            response = self.llm.query(prompt)

            # Parse response
            thought, selected_task_id = self._parse_reasoning(response)

            # Get the actual task
            next_task = state.get_task_by_id(selected_task_id)

            if next_task is None or next_task not in available_tasks:
                # Fallback to first available task
                self.logger.warning(f"Selected task {selected_task_id} not available, using first available")
                next_task = available_tasks[0]
                thought = f"{thought}\n(Fallback: Selected first available task)"

            self.logger.info(f"Reasoned to select task {next_task.id}: {next_task.description}")
            return thought, next_task

        except Exception as e:
            self.logger.error(f"Reasoning failed: {e}. Selecting first available task.")
            task = available_tasks[0]
            thought = f"Reasoning failed, selecting first available task: {task.description}"
            return thought, task

    def _build_reasoning_prompt(self, state: AgentState, available_tasks: List[Task]) -> str:
        """Build prompt for reasoning."""
        completed_count = len(state.get_completed_tasks())
        total_count = len(state.tasks)

        recent_thoughts = state.thought_history[-3:] if len(state.thought_history) > 0 else ["None yet"]
        recent_actions = self._format_recent_actions(state.action_history[-3:]) if state.action_history else "None yet"

        tasks_text = self._format_tasks(available_tasks)

        return f"""You are working on this goal: {state.goal}

Progress so far:
- Iteration: {state.iteration}/{state.max_iterations}
- Completed tasks: {completed_count}/{total_count}
- Recent thoughts: {recent_thoughts}
- Recent actions:
{recent_actions}

Available tasks to work on next:
{tasks_text}

Question: Which task should I work on next and why?

Think step-by-step:
1. What have I accomplished so far?
2. What is blocking me from achieving the goal?
3. Which task will make the most progress toward the goal?
4. Are there any dependencies or risks to consider?

Respond in JSON format:
{{
  "thought": "My reasoning here (2-3 sentences)...",
  "selected_task_id": "task_X",
  "rationale": "Why I chose this task (1-2 sentences)..."
}}

Important: Ensure the JSON is valid."""

    def _format_tasks(self, tasks: List[Task]) -> str:
        """Format tasks for display in prompt."""
        lines = []
        for task in tasks:
            complexity = task.metadata.get("complexity", "unknown")
            tools = task.metadata.get("tools", [])
            deps = f" (depends on: {', '.join(task.dependencies)})" if task.dependencies else ""
            tools_str = f" [tools: {', '.join(tools)}]" if tools else ""
            lines.append(f"- {task.id}: {task.description}{deps}{tools_str} [complexity: {complexity}]")
        return "\n".join(lines) if lines else "No tasks available"

    def _format_recent_actions(self, actions: List[dict]) -> str:
        """Format recent actions for prompt."""
        if not actions:
            return "None yet"

        lines = []
        for action in actions:
            task_id = action.get("task_id", "unknown")
            success = action.get("success", False)
            status = "✓" if success else "✗"
            lines.append(f"  {status} {task_id}")

        return "\n".join(lines)

    def _parse_reasoning(self, response: str) -> Tuple[str, str]:
        """
        Parse reasoning response.

        Args:
            response: LLM response

        Returns:
            (thought, selected_task_id)
        """
        try:
            # Extract JSON
            json_str = self._extract_json(response)
            data = json.loads(json_str)

            thought = data.get("thought", "No thought provided")
            rationale = data.get("rationale", "")
            selected_task_id = data.get("selected_task_id", "")

            # Combine thought and rationale
            full_thought = f"{thought} {rationale}".strip()

            return full_thought, selected_task_id

        except Exception as e:
            self.logger.warning(f"Failed to parse reasoning: {e}")
            # Try to extract task_id from response as fallback
            task_id = self._extract_task_id_fallback(response)
            return response[:200], task_id

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text."""
        # Try to find JSON block in markdown
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        # Try to find JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return text[start:end]

        return text

    def _extract_task_id_fallback(self, text: str) -> str:
        """Extract task_id from text as fallback."""
        # Look for patterns like "task_1", "task_2", etc.
        import re
        match = re.search(r'task_\d+', text)
        return match.group(0) if match else ""
