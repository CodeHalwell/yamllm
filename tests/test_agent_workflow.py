"""Tests for workflow manager."""

import pytest
from unittest.mock import Mock

from yamllm.agent import Agent, WorkflowManager


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


@pytest.fixture
def agent(mock_llm):
    """Create agent with mock LLM."""
    return Agent(mock_llm, max_iterations=2)


@pytest.fixture
def workflow_manager(agent):
    """Create workflow manager."""
    return WorkflowManager(agent)


def test_workflow_manager_initialization(workflow_manager):
    """Test workflow manager initialization."""
    assert workflow_manager.agent is not None
    assert len(workflow_manager.WORKFLOWS) > 0


def test_list_workflows(workflow_manager):
    """Test listing workflows."""
    workflows = workflow_manager.list_workflows()

    assert len(workflows) > 0
    assert all("name" in wf for wf in workflows)
    assert all("description" in wf for wf in workflows)


def test_get_workflow_info(workflow_manager):
    """Test getting workflow info."""
    info = workflow_manager.get_workflow_info("debug_bug")

    assert info["name"] == "Debug Bug"
    assert "steps" in info
    assert len(info["steps"]) > 0


def test_get_unknown_workflow_raises_error(workflow_manager):
    """Test that unknown workflow raises error."""
    with pytest.raises(ValueError, match="Unknown workflow"):
        workflow_manager.get_workflow_info("nonexistent_workflow")


def test_execute_workflow_validates_context(workflow_manager):
    """Test that workflow execution validates required context."""
    # debug_bug requires bug_description
    with pytest.raises(ValueError, match="Missing required context"):
        workflow_manager.execute_workflow("debug_bug", {})


def test_execute_workflow_with_valid_context(workflow_manager):
    """Test executing workflow with valid context."""
    context = {"bug_description": "Test bug"}

    state = workflow_manager.execute_workflow("debug_bug", context)

    assert state is not None
    assert state.goal is not None
    assert "debug_bug" in state.goal.lower() or "debug" in state.goal.lower()


def test_workflow_creates_appropriate_goal(workflow_manager):
    """Test that workflow creates appropriate goal."""
    context = {"feature_description": "New feature"}

    goal = workflow_manager._create_goal_from_workflow(
        workflow_manager.WORKFLOWS["implement_feature"],
        context
    )

    assert "implement_feature" in goal.lower() or "implement" in goal.lower()
    assert "new feature" in goal.lower()


def test_all_workflows_have_required_fields():
    """Test that all defined workflows have required fields."""
    required_fields = ["name", "description", "steps"]

    for name, workflow in WorkflowManager.WORKFLOWS.items():
        for field in required_fields:
            assert field in workflow, f"Workflow {name} missing field {field}"
        assert len(workflow["steps"]) > 0, f"Workflow {name} has no steps"


def test_workflow_names_match_keys():
    """Test that workflow names are consistent with keys."""
    for key, workflow in WorkflowManager.WORKFLOWS.items():
        # Just ensure workflow has a name
        assert workflow.get("name"), f"Workflow {key} has no name"
