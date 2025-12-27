"""Data models for the agentic system."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import uuid


class TaskStatus(Enum):
    """Status of a task in the agent workflow."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class Task:
    """Represents a single task in the agent workflow."""

    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, description: str, dependencies: Optional[List[str]] = None, **metadata) -> "Task":
        """Create a new task with auto-generated ID."""
        return cls(
            id=str(uuid.uuid4())[:8],
            description=description,
            dependencies=dependencies or [],
            metadata=metadata
        )


@dataclass
class AgentState:
    """Current state of the agent execution."""

    goal: str
    tasks: List[Task] = field(default_factory=list)
    current_task_id: Optional[str] = None
    iteration: int = 0
    max_iterations: int = 10
    thought_history: List[str] = field(default_factory=list)
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    completed: bool = False
    success: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """Get a task by its ID."""
        return next((t for t in self.tasks if t.id == task_id), None)

    def get_completed_tasks(self) -> List[Task]:
        """Get all completed tasks."""
        return [t for t in self.tasks if t.status == TaskStatus.COMPLETED]

    def get_pending_tasks(self) -> List[Task]:
        """Get all pending tasks."""
        return [t for t in self.tasks if t.status == TaskStatus.PENDING]

    def get_available_tasks(self) -> List[Task]:
        """Get tasks that can be executed (dependencies met, not completed)."""
        available = []
        completed_ids = {t.id for t in self.tasks if t.status == TaskStatus.COMPLETED}

        for task in self.tasks:
            if task.status in [TaskStatus.COMPLETED, TaskStatus.IN_PROGRESS]:
                continue

            # Check if all dependencies are completed
            deps_met = all(dep_id in completed_ids for dep_id in task.dependencies)

            if deps_met:
                available.append(task)

        return available

    def get_progress(self) -> float:
        """Calculate progress as percentage of completed tasks."""
        if not self.tasks:
            return 0.0
        completed = len(self.get_completed_tasks())
        return (completed / len(self.tasks)) * 100

    def add_thought(self, thought: str) -> None:
        """Add a thought to the history."""
        self.thought_history.append(thought)

    def add_action(self, action: Dict[str, Any]) -> None:
        """Add an action to the history."""
        self.action_history.append(action)


@dataclass
class ActionResult:
    """Result of an action execution."""

    task_id: str
    success: bool
    response: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    execution_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "success": self.success,
            "response": self.response,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "error": self.error,
            "execution_time": self.execution_time
        }


@dataclass
class Observation:
    """Observations from action results."""

    success_assessment: bool
    learned: str
    unblocked_tasks: List[str] = field(default_factory=list)
    progress_made: str = ""
    plan_adjustments: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Observation":
        """Create from dictionary."""
        return cls(
            success_assessment=data.get("success_assessment", False),
            learned=data.get("learned", ""),
            unblocked_tasks=data.get("unblocked_tasks", []),
            progress_made=data.get("progress_made", ""),
            plan_adjustments=data.get("plan_adjustments", "")
        )
