"""Tests for agent models."""

import pytest
from yamllm.agent.models import Task, TaskStatus, AgentState, ActionResult, Observation


def test_task_creation():
    """Test task creation."""
    task = Task.create("Test task description")

    assert task.id is not None
    assert task.description == "Test task description"
    assert task.status == TaskStatus.PENDING
    assert task.dependencies == []
    assert task.result is None
    assert task.error is None


def test_task_with_dependencies():
    """Test task with dependencies."""
    task = Task.create("Task with deps", dependencies=["task_1", "task_2"])

    assert len(task.dependencies) == 2
    assert "task_1" in task.dependencies
    assert "task_2" in task.dependencies


def test_agent_state_creation():
    """Test agent state creation."""
    state = AgentState(goal="Test goal")

    assert state.goal == "Test goal"
    assert state.tasks == []
    assert state.iteration == 0
    assert not state.completed
    assert not state.success


def test_agent_state_get_task_by_id():
    """Test getting task by ID."""
    task1 = Task.create("Task 1")
    task2 = Task.create("Task 2")

    state = AgentState(goal="Test", tasks=[task1, task2])

    found = state.get_task_by_id(task1.id)
    assert found == task1

    not_found = state.get_task_by_id("nonexistent")
    assert not_found is None


def test_agent_state_get_completed_tasks():
    """Test getting completed tasks."""
    task1 = Task.create("Task 1")
    task1.status = TaskStatus.COMPLETED

    task2 = Task.create("Task 2")
    task2.status = TaskStatus.PENDING

    state = AgentState(goal="Test", tasks=[task1, task2])

    completed = state.get_completed_tasks()
    assert len(completed) == 1
    assert completed[0] == task1


def test_agent_state_get_available_tasks():
    """Test getting available tasks (dependencies met)."""
    task1 = Task.create("Task 1")
    task1.status = TaskStatus.COMPLETED

    task2 = Task.create("Task 2", dependencies=[task1.id])

    task3 = Task.create("Task 3", dependencies=["nonexistent"])

    state = AgentState(goal="Test", tasks=[task1, task2, task3])

    available = state.get_available_tasks()

    # task1 is completed, so not available
    # task2's dependency is met, so it's available
    # task3's dependency not met, so not available
    assert len(available) == 1
    assert available[0] == task2


def test_agent_state_progress():
    """Test progress calculation."""
    task1 = Task.create("Task 1")
    task1.status = TaskStatus.COMPLETED

    task2 = Task.create("Task 2")
    task2.status = TaskStatus.COMPLETED

    task3 = Task.create("Task 3")
    task3.status = TaskStatus.PENDING

    state = AgentState(goal="Test", tasks=[task1, task2, task3])

    progress = state.get_progress()
    assert progress == pytest.approx(66.67, rel=0.1)


def test_action_result_to_dict():
    """Test action result serialization."""
    result = ActionResult(
        task_id="task_1",
        success=True,
        response="Test response",
        execution_time=1.5
    )

    data = result.to_dict()

    assert data["task_id"] == "task_1"
    assert data["success"] is True
    assert data["response"] == "Test response"
    assert data["execution_time"] == 1.5


def test_observation_from_dict():
    """Test observation deserialization."""
    data = {
        "success_assessment": True,
        "learned": "Something learned",
        "unblocked_tasks": ["task_2"],
        "progress_made": "Good progress",
        "plan_adjustments": "No adjustments"
    }

    obs = Observation.from_dict(data)

    assert obs.success_assessment is True
    assert obs.learned == "Something learned"
    assert len(obs.unblocked_tasks) == 1
    assert obs.progress_made == "Good progress"
