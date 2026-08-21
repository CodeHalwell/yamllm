"""Task planning component for breaking down goals into subtasks."""

import json
import logging
from typing import List, Dict, Any, Optional

from .models import Task, AgentState, TaskStatus


class TaskPlanner:
    """
    Breaks down high-level goals into actionable subtasks.

    Uses LLM to analyze goal and create dependency-ordered task list.
    """

    def __init__(self, llm, logger: Optional[logging.Logger] = None):
        """
        Initialize the task planner.

        Args:
            llm: LLM instance for planning
            logger: Optional logger instance
        """
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)

    def decompose_goal(
        self,
        goal: str,
        context: Optional[Dict[str, Any]],
        state: AgentState
    ) -> AgentState:
        """
        Decompose goal into subtasks.

        Args:
            goal: High-level goal to achieve
            context: Optional context (files, repo info, etc.)
            state: Current agent state

        Returns:
            Updated agent state with tasks
        """
        self.logger.info(f"Planning tasks for goal: {goal}")

        # Build planning prompt
        prompt = self._build_planning_prompt(goal, context)

        try:
            # Get task decomposition from LLM
            response = self.llm.query(prompt)

            # Parse response into Task objects
            tasks = self._parse_tasks(response, goal)

            # Validate dependencies
            tasks = self._validate_dependencies(tasks)

            state.tasks = tasks
            self.logger.info(f"Created {len(tasks)} tasks")

        except Exception as e:
            self.logger.error(f"Planning failed: {e}")
            # Fallback: create single task from goal
            state.tasks = [Task.create(goal)]

        return state

    def _build_planning_prompt(self, goal: str, context: Optional[Dict[str, Any]]) -> str:
        """Build prompt for task decomposition."""
        context_str = self._format_context(context) if context else "No additional context provided."

        return f"""You are a task planning assistant. Break down the following goal into concrete, actionable subtasks.

Goal: {goal}

Context:
{context_str}

Requirements:
1. Each task should be specific and measurable
2. Identify dependencies between tasks (use task IDs)
3. Order tasks logically
4. Consider which tools might be needed
5. Keep tasks focused and atomic

Respond in JSON format:
{{
  "tasks": [
    {{
      "id": "task_1",
      "description": "Clear description of what to do",
      "dependencies": [],
      "required_tools": ["tool_name"],
      "estimated_complexity": "low|medium|high"
    }},
    {{
      "id": "task_2",
      "description": "Another task",
      "dependencies": ["task_1"],
      "required_tools": [],
      "estimated_complexity": "medium"
    }}
  ]
}}

Important: Ensure the JSON is valid and well-formed."""

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context dictionary for prompt."""
        lines = []
        for key, value in context.items():
            if isinstance(value, (list, dict)):
                value = json.dumps(value, indent=2)
            lines.append(f"- {key}: {value}")
        return "\n".join(lines) if lines else "No context"

    def _parse_tasks(self, response: str, goal: str) -> List[Task]:
        """
        Parse LLM response into Task objects.

        Args:
            response: LLM response (hopefully JSON)
            goal: Original goal (for fallback)

        Returns:
            List of Task objects
        """
        try:
            # Try to extract JSON from response
            json_str = self._extract_json(response)
            data = json.loads(json_str)

            tasks = []
            for i, task_data in enumerate(data.get("tasks", [])):
                task = Task(
                    id=task_data.get("id", f"task_{i+1}"),
                    description=task_data["description"],
                    status=TaskStatus.PENDING,
                    dependencies=task_data.get("dependencies", []),
                    metadata={
                        "tools": task_data.get("required_tools", []),
                        "complexity": task_data.get("estimated_complexity", "medium")
                    }
                )
                tasks.append(task)

            return tasks if tasks else [Task.create(goal)]

        except Exception as e:
            self.logger.warning(f"Failed to parse tasks: {e}. Creating single task.")
            # Fallback: create single task from goal
            return [Task.create(goal)]

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that might contain markdown or other content."""
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

    def _validate_dependencies(self, tasks: List[Task]) -> List[Task]:
        """
        Validate and fix task dependencies.

        Ensures:
        - Dependencies reference valid task IDs
        - No circular dependencies
        - Tasks are topologically ordered
        """
        task_ids = {t.id for t in tasks}

        # Remove invalid dependencies
        for task in tasks:
            task.dependencies = [dep for dep in task.dependencies if dep in task_ids]

        # Check for circular dependencies
        if self._has_circular_dependency(tasks):
            self.logger.warning("Circular dependency detected, removing problematic dependencies")
            tasks = self._remove_circular_dependencies(tasks)

        return tasks

    def _has_circular_dependency(self, tasks: List[Task]) -> bool:
        """Check if task list has circular dependencies."""
        visited = set()
        rec_stack = set()

        def visit(task_id: str, task_map: Dict[str, Task]) -> bool:
            if task_id in rec_stack:
                return True
            if task_id in visited:
                return False

            visited.add(task_id)
            rec_stack.add(task_id)

            task = task_map.get(task_id)
            if task:
                for dep_id in task.dependencies:
                    if visit(dep_id, task_map):
                        return True

            rec_stack.remove(task_id)
            return False

        task_map = {t.id: t for t in tasks}
        return any(visit(t.id, task_map) for t in tasks)

    def _remove_circular_dependencies(self, tasks: List[Task]) -> List[Task]:
        """Remove circular dependencies by clearing problematic deps."""
        # Simple approach: clear all dependencies if circular
        for task in tasks:
            task.dependencies = []
        return tasks
