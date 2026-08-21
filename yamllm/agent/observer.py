"""Observation component for interpreting action results."""

import json
import logging
from typing import Optional

from .models import AgentState, ActionResult, Observation, TaskStatus


class Observer:
    """
    Observation component - interprets results.

    Implements the 'Observe' part of ReAct.
    """

    def __init__(self, llm, logger: Optional[logging.Logger] = None):
        """
        Initialize the observer.

        Args:
            llm: LLM instance for observation
            logger: Optional logger instance
        """
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)

    def observe(self, action_result: ActionResult, state: AgentState) -> AgentState:
        """
        Observe action result and update state.

        Analyzes:
        - Did the action succeed?
        - What was learned?
        - Does this unblock other tasks?
        - Are we closer to the goal?

        Args:
            action_result: Result of action execution
            state: Current agent state

        Returns:
            Updated agent state
        """
        self.logger.info(f"Observing result of task {action_result.task_id}")

        # For simple success/failure, we can skip LLM observation
        if not action_result.success:
            self.logger.warning(f"Task {action_result.task_id} failed: {action_result.error}")
            # Don't ask LLM to interpret obvious failures
            observations = Observation(
                success_assessment=False,
                learned=f"Task failed with error: {action_result.error}",
                unblocked_tasks=[],
                progress_made="No progress - task failed",
                plan_adjustments="May need to retry or adjust approach"
            )
        else:
            # Build observation prompt
            prompt = self._build_observation_prompt(action_result, state)

            try:
                # Get LLM's interpretation
                response = self.llm.query(prompt)

                # Parse observations
                observations = self._parse_observations(response)

                self.logger.info(f"Observation: {observations.learned[:100]}...")

            except Exception as e:
                self.logger.error(f"Observation failed: {e}")
                # Fallback: basic success observation
                observations = Observation(
                    success_assessment=True,
                    learned="Task completed successfully",
                    unblocked_tasks=[],
                    progress_made="Made progress on goal",
                    plan_adjustments=""
                )

        # Update state based on observations
        state = self._update_state(observations, state, action_result)

        return state

    def _build_observation_prompt(self, action_result: ActionResult, state: AgentState) -> str:
        """Build prompt for observation."""
        tool_names = [tc.get("function", {}).get("name", "unknown") for tc in action_result.tool_calls]

        return f"""You are observing the result of an action taken to achieve this goal: {state.goal}

Action taken:
- Task ID: {action_result.task_id}
- Task: {self._get_task_description(action_result.task_id, state)}

Result:
- Success: {action_result.success}
- Tools used: {tool_names if tool_names else 'None'}
- Output: {action_result.response[:500] if action_result.response else 'N/A'}
- Error: {action_result.error or 'None'}
- Execution time: {action_result.execution_time:.2f}s

Remaining tasks:
{self._format_remaining_tasks(state)}

Questions to answer:
1. Was this action successful in completing its task?
2. What did we learn or discover?
3. Does this unblock any other tasks?
4. Are we closer to achieving the overall goal?
5. Do we need to adjust our plan?

Respond in JSON format:
{{
  "success_assessment": true/false,
  "learned": "What we learned (1-2 sentences)...",
  "unblocked_tasks": ["task_id"],
  "progress_made": "Description of progress (1 sentence)...",
  "plan_adjustments": "Any needed adjustments (or empty string)..."
}}

Important: Keep responses concise. Ensure valid JSON."""

    def _get_task_description(self, task_id: str, state: AgentState) -> str:
        """Get task description by ID."""
        task = state.get_task_by_id(task_id)
        return task.description if task else "Unknown task"

    def _format_remaining_tasks(self, state: AgentState) -> str:
        """Format remaining tasks."""
        pending = state.get_pending_tasks()
        if not pending:
            return "No remaining tasks"

        lines = []
        for task in pending[:5]:  # Show first 5
            lines.append(f"- {task.id}: {task.description}")

        if len(pending) > 5:
            lines.append(f"... and {len(pending) - 5} more")

        return "\n".join(lines)

    def _parse_observations(self, response: str) -> Observation:
        """Parse observation response."""
        try:
            # Extract JSON
            json_str = self._extract_json(response)
            data = json.loads(json_str)

            return Observation.from_dict(data)

        except Exception as e:
            self.logger.warning(f"Failed to parse observations: {e}")
            # Fallback
            return Observation(
                success_assessment=True,
                learned="Task completed",
                unblocked_tasks=[],
                progress_made="Made progress",
                plan_adjustments=""
            )

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

    def _update_state(
        self,
        observations: Observation,
        state: AgentState,
        action_result: ActionResult
    ) -> AgentState:
        """Update state based on observations."""

        # Unblock dependent tasks if action succeeded
        if observations.success_assessment:
            task_id = action_result.task_id
            for task in state.tasks:
                if task_id in task.dependencies and task.status == TaskStatus.BLOCKED:
                    task.status = TaskStatus.PENDING
                    self.logger.info(f"Unblocked task {task.id}")

        # Add learned info to state metadata
        if observations.learned:
            if "learnings" not in state.metadata:
                state.metadata["learnings"] = []
            state.metadata["learnings"].append(observations.learned)

        # Store observation in metadata
        if "observations" not in state.metadata:
            state.metadata["observations"] = []
        state.metadata["observations"].append({
            "task_id": action_result.task_id,
            "learned": observations.learned,
            "progress": observations.progress_made
        })

        return state
