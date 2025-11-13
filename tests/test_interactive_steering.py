"""Tests for interactive agent steering."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from yamllm.agent.interactive_steering import (
    InteractiveSteering,
    InteractiveAgent,
    SteeringAction,
    SteeringDecision,
    SteeringPoint
)


def test_steering_initialization():
    """Test steering controller initialization."""
    steering = InteractiveSteering(auto_approve=True)

    assert steering.auto_approve is True
    assert steering.pause_before_action is True
    assert steering.is_paused is False
    assert steering.should_stop is False
    assert len(steering.watchpoints) == 0
    assert len(steering.decision_history) == 0


def test_steering_auto_approve():
    """Test auto-approve mode."""
    steering = InteractiveSteering(auto_approve=True)

    point = SteeringPoint(
        iteration=1,
        thought="Test thought",
        planned_action={"task": "test"},
        current_state=None,
        context={}
    )

    decision = steering.request_decision(point)

    assert decision.action == SteeringAction.APPROVE
    assert len(steering.decision_history) == 1


def test_watchpoint_add_and_clear():
    """Test adding and clearing watchpoints."""
    steering = InteractiveSteering()

    # Add watchpoint
    steering.add_watchpoint(lambda sp: "delete" in sp.thought.lower())

    assert len(steering.watchpoints) == 1

    # Clear watchpoints
    steering.clear_watchpoints()

    assert len(steering.watchpoints) == 0


def test_watchpoint_trigger():
    """Test watchpoint triggering."""
    steering = InteractiveSteering(auto_approve=True)

    # Add watchpoint for dangerous operations
    steering.add_watchpoint(lambda sp: "delete" in sp.thought.lower())

    # Point without trigger
    point1 = SteeringPoint(
        iteration=1,
        thought="Read file",
        planned_action={},
        current_state=None,
        context={}
    )

    assert not steering.check_watchpoints(point1)

    # Point with trigger
    point2 = SteeringPoint(
        iteration=2,
        thought="Delete all files",
        planned_action={},
        current_state=None,
        context={}
    )

    assert steering.check_watchpoints(point2)


def test_decision_history():
    """Test decision history tracking."""
    steering = InteractiveSteering(auto_approve=True)

    # Make several decisions
    for i in range(3):
        point = SteeringPoint(
            iteration=i+1,
            thought=f"Thought {i}",
            planned_action={},
            current_state=None,
            context={}
        )
        steering.request_decision(point)

    assert len(steering.decision_history) == 3
    assert all(d.action == SteeringAction.APPROVE for d in steering.decision_history)


def test_steering_summary():
    """Test getting steering summary."""
    steering = InteractiveSteering(auto_approve=True)

    # Simulate decisions
    steering.decision_history = [
        SteeringDecision(action=SteeringAction.APPROVE),
        SteeringDecision(action=SteeringAction.APPROVE),
        SteeringDecision(action=SteeringAction.REJECT, feedback="test"),
        SteeringDecision(action=SteeringAction.MODIFY, feedback="test"),
    ]

    summary = steering.get_summary()

    assert summary["total_decisions"] == 4
    assert summary["action_counts"]["approve"] == 2
    assert summary["action_counts"]["reject"] == 1
    assert summary["action_counts"]["modify"] == 1
    assert summary["auto_approved"] is True


def test_steering_point_dataclass():
    """Test SteeringPoint dataclass."""
    point = SteeringPoint(
        iteration=1,
        thought="Test reasoning",
        planned_action={"task_id": "123", "action": "test"},
        current_state=None,
        context={"key": "value"}
    )

    assert point.iteration == 1
    assert point.thought == "Test reasoning"
    assert point.planned_action["task_id"] == "123"
    assert point.context["key"] == "value"


def test_steering_decision_dataclass():
    """Test SteeringDecision dataclass."""
    decision = SteeringDecision(
        action=SteeringAction.MODIFY,
        feedback="Please be more careful",
        modified_task="New task description"
    )

    assert decision.action == SteeringAction.MODIFY
    assert decision.feedback == "Please be more careful"
    assert decision.modified_task == "New task description"
    assert not decision.auto_approve_remaining


def test_steering_action_enum():
    """Test SteeringAction enum values."""
    assert SteeringAction.APPROVE.value == "approve"
    assert SteeringAction.REJECT.value == "reject"
    assert SteeringAction.MODIFY.value == "modify"
    assert SteeringAction.PAUSE.value == "pause"
    assert SteeringAction.SKIP.value == "skip"
    assert SteeringAction.STOP.value == "stop"
    assert SteeringAction.AUTO.value == "auto"


def test_interactive_agent_initialization():
    """Test interactive agent initialization."""
    mock_agent = Mock()
    mock_agent.max_iterations = 10

    interactive = InteractiveAgent(
        agent=mock_agent,
        pause_before_action=True,
        auto_approve=False
    )

    assert interactive.agent == mock_agent
    assert interactive.steering is not None
    assert interactive.steering.pause_before_action is True
    assert interactive.steering.auto_approve is False


def test_interactive_agent_with_custom_steering():
    """Test interactive agent with custom steering controller."""
    mock_agent = Mock()
    custom_steering = InteractiveSteering(auto_approve=True)

    interactive = InteractiveAgent(agent=mock_agent, steering=custom_steering)

    assert interactive.steering is custom_steering
    assert interactive.steering.auto_approve is True


def test_watchpoint_error_handling():
    """Test that watchpoint errors are handled gracefully."""
    steering = InteractiveSteering()

    # Add a watchpoint that raises an error
    def bad_watchpoint(sp):
        raise ValueError("Test error")

    steering.add_watchpoint(bad_watchpoint)

    point = SteeringPoint(
        iteration=1,
        thought="Test",
        planned_action={},
        current_state=None,
        context={}
    )

    # Should not raise, should return False
    result = steering.check_watchpoints(point)
    assert result is False


def test_multiple_watchpoints():
    """Test multiple watchpoints."""
    steering = InteractiveSteering(auto_approve=True)

    # Add multiple watchpoints
    steering.add_watchpoint(lambda sp: "delete" in sp.thought.lower())
    steering.add_watchpoint(lambda sp: "remove" in sp.thought.lower())
    steering.add_watchpoint(lambda sp: sp.iteration > 5)

    assert len(steering.watchpoints) == 3

    # Test first watchpoint
    point1 = SteeringPoint(
        iteration=1,
        thought="Delete file",
        planned_action={},
        current_state=None,
        context={}
    )
    assert steering.check_watchpoints(point1)

    # Test second watchpoint
    point2 = SteeringPoint(
        iteration=2,
        thought="Remove directory",
        planned_action={},
        current_state=None,
        context={}
    )
    assert steering.check_watchpoints(point2)

    # Test third watchpoint
    point3 = SteeringPoint(
        iteration=10,
        thought="Normal operation",
        planned_action={},
        current_state=None,
        context={}
    )
    assert steering.check_watchpoints(point3)

    # Test no trigger
    point4 = SteeringPoint(
        iteration=1,
        thought="Normal operation",
        planned_action={},
        current_state=None,
        context={}
    )
    assert not steering.check_watchpoints(point4)


def test_steering_state_transitions():
    """Test steering state transitions."""
    steering = InteractiveSteering()

    # Initially not paused, not stopped
    assert not steering.is_paused
    assert not steering.should_stop

    # Pause
    steering.is_paused = True
    assert steering.is_paused

    # Stop
    steering.should_stop = True
    assert steering.should_stop


def test_decision_with_feedback():
    """Test decisions with feedback."""
    decision1 = SteeringDecision(
        action=SteeringAction.REJECT,
        feedback="This is too risky"
    )

    assert decision1.feedback == "This is too risky"

    decision2 = SteeringDecision(
        action=SteeringAction.MODIFY,
        feedback="Use a safer approach",
        modified_task="Updated task description"
    )

    assert decision2.feedback == "Use a safer approach"
    assert decision2.modified_task == "Updated task description"


def test_steering_with_context():
    """Test steering with rich context."""
    steering = InteractiveSteering(auto_approve=True)

    point = SteeringPoint(
        iteration=5,
        thought="Analyzing code for vulnerabilities",
        planned_action={
            "task_id": "task_123",
            "tool": "code_analyzer",
            "parameters": {"depth": "full"}
        },
        current_state=Mock(
            get_progress=lambda: 0.5,
            get_completed_tasks=lambda: [1, 2, 3]
        ),
        context={
            "completed_tasks": 3,
            "total_tasks": 6,
            "current_file": "security.py",
            "severity": "high"
        }
    )

    decision = steering.request_decision(point)

    assert decision.action == SteeringAction.APPROVE
    assert point.context["severity"] == "high"
    assert len(point.context) == 4
