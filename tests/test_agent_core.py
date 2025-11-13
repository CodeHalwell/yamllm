"""Tests for agent core functionality."""

import pytest
from unittest.mock import Mock, MagicMock, patch

from yamllm.agent import Agent, SimpleAgent
from yamllm.agent.models import TaskStatus


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    llm = Mock()
    llm.query = Mock(return_value="Mock response")
    llm.get_completion_with_tools = Mock(return_value={
        "content": "Mock response",
        "tool_calls": [],
        "tool_results": []
    })
    return llm


def test_agent_initialization(mock_llm):
    """Test agent initialization."""
    agent = Agent(mock_llm, max_iterations=5)

    assert agent.llm == mock_llm
    assert agent.max_iterations == 5
    assert agent.enable_planning is True
    assert agent.planner is not None
    assert agent.reasoner is not None
    assert agent.actor is not None
    assert agent.observer is not None


def test_simple_agent_initialization(mock_llm):
    """Test simple agent initialization."""
    agent = SimpleAgent(mock_llm, max_iterations=3)

    assert agent.llm == mock_llm
    assert agent.max_iterations == 3
    assert agent.enable_planning is False
    assert agent.enable_reflection is False


def test_agent_execute_creates_state(mock_llm):
    """Test that agent.execute creates proper state."""
    # Mock planner to return tasks
    with patch('yamllm.agent.planner.TaskPlanner.decompose_goal') as mock_decompose:
        def add_tasks(goal, context, state):
            from yamllm.agent.models import Task
            state.tasks = [Task.create("Test task")]
            return state

        mock_decompose.side_effect = add_tasks

        agent = Agent(mock_llm, max_iterations=2)
        state = agent.execute("Test goal")

        assert state is not None
        assert state.goal == "Test goal"
        assert state.completed is True


def test_simple_agent_execute(mock_llm):
    """Test simple agent execution."""
    agent = SimpleAgent(mock_llm)

    state = agent.execute("Simple task")

    assert state.goal == "Simple task"
    assert len(state.tasks) == 1
    assert state.completed is True


def test_agent_max_iterations_respected(mock_llm):
    """Test that agent respects max iterations."""
    # Mock to keep returning pending tasks
    with patch('yamllm.agent.planner.TaskPlanner.decompose_goal') as mock_decompose:
        def add_many_tasks(goal, context, state):
            from yamllm.agent.models import Task
            # Create more tasks than iterations
            state.tasks = [Task.create(f"Task {i}") for i in range(10)]
            return state

        mock_decompose.side_effect = add_many_tasks

        agent = Agent(mock_llm, max_iterations=3)
        state = agent.execute("Test goal")

        # Should stop at max iterations
        assert state.iteration <= agent.max_iterations
        assert state.completed is True


def test_agent_progress_callback(mock_llm):
    """Test progress callback is called."""
    callback_calls = []

    def callback(state):
        callback_calls.append(state.iteration)

    agent = SimpleAgent(mock_llm)
    agent.progress_callback = callback

    state = agent.execute("Test goal")

    # Callback should have been called
    assert len(callback_calls) > 0


def test_agent_handles_execution_error(mock_llm):
    """Test agent handles execution errors gracefully."""
    # Make LLM raise error
    mock_llm.query.side_effect = Exception("Test error")

    agent = SimpleAgent(mock_llm)
    state = agent.execute("Test goal")

    assert state.completed is True
    assert state.success is False
    assert state.error is not None


def test_agent_completion_status_calculation(mock_llm):
    """Test completion status calculation."""
    with patch('yamllm.agent.planner.TaskPlanner.decompose_goal') as mock_decompose:
        def add_completed_tasks(goal, context, state):
            from yamllm.agent.models import Task
            task1 = Task.create("Task 1")
            task1.status = TaskStatus.COMPLETED
            task2 = Task.create("Task 2")
            task2.status = TaskStatus.COMPLETED
            state.tasks = [task1, task2]
            return state

        mock_decompose.side_effect = add_completed_tasks

        agent = Agent(mock_llm, max_iterations=1)
        state = agent.execute("Test goal")

        assert state.completed is True
        assert state.success is True
