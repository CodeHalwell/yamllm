"""Tests for the modern AgentHarness engine."""

import json

import pytest
from unittest.mock import Mock

from yamllm.agent.events import AgentEvent, EventKind
from yamllm.agent.harness import AgentHarness, ApprovalPolicy, load_checkpoint
from yamllm.agent.interactive_steering import SteeringAction, SteeringDecision
from yamllm.agent.models import AgentState, Task, TaskStatus
from yamllm.agent.parsing import extract_json_block, parse_json_response


def make_llm(plan_tasks=None):
    """Create a mock LLM that returns a plan on the first query."""
    plan = {
        "tasks": (
            plan_tasks
            if plan_tasks is not None
            else [
                {"id": "task_1", "description": "First task", "dependencies": []},
                {
                    "id": "task_2",
                    "description": "Second task",
                    "dependencies": ["task_1"],
                },
            ]
        )
    }
    llm = Mock()
    llm.query = Mock(side_effect=[json.dumps(plan)] + ["{}"] * 100)
    llm.get_completion_with_tools = Mock(
        return_value={"content": "Done", "tool_calls": [], "tool_results": []}
    )
    return llm


def collect_events(harness):
    events = []
    harness.add_listener(events.append)
    return events


def kinds(events):
    return [e.kind for e in events]


def test_successful_run_emits_lifecycle_events():
    harness = AgentHarness(make_llm(), max_iterations=5)
    events = collect_events(harness)

    state = harness.run("Do the thing")

    assert state.completed and state.success
    seen = kinds(events)
    assert seen[0] == EventKind.RUN_STARTED
    assert EventKind.PLAN_CREATED in seen
    assert EventKind.ITERATION_STARTED in seen
    assert EventKind.THOUGHT in seen
    assert EventKind.ACTION_STARTED in seen
    assert EventKind.ACTION_FINISHED in seen
    assert seen[-1] == EventKind.RUN_FINISHED


def test_events_serialise_to_json():
    harness = AgentHarness(make_llm(), max_iterations=5)
    events = collect_events(harness)

    harness.run("Do the thing")

    for event in events:
        line = event.to_json()
        parsed = json.loads(line)
        assert parsed["kind"] == event.kind.value
        assert "payload" in parsed


def test_run_finished_payload_summarises_run():
    harness = AgentHarness(make_llm(), max_iterations=5)
    events = collect_events(harness)

    harness.run("Do the thing")

    finished = [e for e in events if e.kind == EventKind.RUN_FINISHED][-1]
    assert finished.payload["success"] is True
    assert finished.payload["tasks_total"] == 2
    assert finished.payload["tasks_completed"] == 2


def test_wall_time_budget_stops_run():
    harness = AgentHarness(make_llm(), max_iterations=50, max_wall_time=0.0)
    events = collect_events(harness)

    state = harness.run("Do the thing")

    assert state.completed and not state.success
    assert "Wall-time budget exceeded" in (state.error or "")
    assert EventKind.BUDGET_EXCEEDED in kinds(events)


def test_consecutive_failures_budget():
    llm = make_llm(
        plan_tasks=[
            {"id": f"task_{i}", "description": f"Task {i}", "dependencies": []}
            for i in range(1, 6)
        ]
    )
    llm.get_completion_with_tools = Mock(side_effect=Exception("boom"))
    harness = AgentHarness(llm, max_iterations=50, max_consecutive_failures=2)
    events = collect_events(harness)

    state = harness.run("Do the thing")

    assert state.completed and not state.success
    assert EventKind.BUDGET_EXCEEDED in kinds(events)


def test_request_stop_cancels_run():
    harness = AgentHarness(make_llm(), max_iterations=5)

    # Stop as soon as the first action starts
    def stop_on_action(event: AgentEvent) -> None:
        if event.kind == EventKind.ACTION_FINISHED:
            harness.request_stop()

    harness.add_listener(stop_on_action)
    state = harness.run("Do the thing")

    assert state.completed and not state.success
    assert state.error == "Stopped by user"


def test_approval_always_uses_decision_provider():
    decisions = []

    def approve(point):
        decisions.append(point)
        return SteeringDecision(action=SteeringAction.APPROVE)

    harness = AgentHarness(
        make_llm(),
        max_iterations=5,
        approval_policy=ApprovalPolicy.ALWAYS,
        decision_provider=approve,
    )
    events = collect_events(harness)

    state = harness.run("Do the thing")

    assert state.success
    assert len(decisions) == 2  # one per task
    assert EventKind.APPROVAL_REQUESTED in kinds(events)
    assert EventKind.DECISION in kinds(events)


def test_reject_decision_fails_task():
    def reject(point):
        return SteeringDecision(action=SteeringAction.REJECT, feedback="nope")

    harness = AgentHarness(
        make_llm(
            plan_tasks=[{"id": "task_1", "description": "Only", "dependencies": []}]
        ),
        max_iterations=5,
        approval_policy=ApprovalPolicy.ALWAYS,
        decision_provider=reject,
    )

    state = harness.run("Do the thing")

    assert state.completed and not state.success
    assert state.tasks[0].status == TaskStatus.FAILED
    assert state.tasks[0].error == "nope"


def test_rejected_task_is_never_executed():
    llm = make_llm(
        plan_tasks=[
            {"id": "task_1", "description": "Risky", "dependencies": []},
            {"id": "task_2", "description": "Safe", "dependencies": []},
        ]
    )
    decisions = iter(
        [
            SteeringDecision(action=SteeringAction.REJECT, feedback="no"),
            SteeringDecision(action=SteeringAction.APPROVE),
        ]
    )

    harness = AgentHarness(
        llm,
        max_iterations=10,
        approval_policy=ApprovalPolicy.ALWAYS,
        decision_provider=lambda point: next(decisions),
    )

    state = harness.run("Do the thing")

    # Only the approved task ran; the rejected one stayed terminal
    assert llm.get_completion_with_tools.call_count == 1
    rejected = state.get_task_by_id("task_1")
    assert rejected.status == TaskStatus.FAILED


def test_stopped_checkpoint_is_resumable(tmp_path):
    def stop(point):
        return SteeringDecision(action=SteeringAction.STOP)

    harness = AgentHarness(
        make_llm(),
        max_iterations=5,
        approval_policy=ApprovalPolicy.ALWAYS,
        decision_provider=stop,
        checkpoint_dir=str(tmp_path),
    )
    state = harness.run("Do the thing")
    assert state.completed and not state.success

    checkpoint = load_checkpoint(state.metadata["checkpoint_path"])
    resumed = AgentHarness(make_llm(), max_iterations=5).run(
        "ignored", initial_state=checkpoint
    )

    assert resumed.completed and resumed.success


def test_stop_decision_ends_run():
    def stop(point):
        return SteeringDecision(action=SteeringAction.STOP)

    harness = AgentHarness(
        make_llm(),
        max_iterations=5,
        approval_policy=ApprovalPolicy.ALWAYS,
        decision_provider=stop,
    )

    state = harness.run("Do the thing")

    assert state.completed and not state.success


def test_auto_decision_stops_prompting():
    decisions = []

    def auto(point):
        decisions.append(point)
        return SteeringDecision(action=SteeringAction.AUTO)

    harness = AgentHarness(
        make_llm(),
        max_iterations=5,
        approval_policy=ApprovalPolicy.ALWAYS,
        decision_provider=auto,
    )

    state = harness.run("Do the thing")

    assert state.success
    assert len(decisions) == 1  # remaining actions auto-approved


def test_on_watchpoint_policy_only_gates_matches():
    decisions = []

    def approve(point):
        decisions.append(point)
        return SteeringDecision(action=SteeringAction.APPROVE)

    harness = AgentHarness(
        make_llm(
            plan_tasks=[
                {"id": "task_1", "description": "Read a file", "dependencies": []},
                {
                    "id": "task_2",
                    "description": "Delete everything",
                    "dependencies": [],
                },
            ]
        ),
        max_iterations=5,
        approval_policy=ApprovalPolicy.ON_WATCHPOINT,
        decision_provider=approve,
    )
    harness.add_watchpoint(
        lambda sp: "delete" in sp.planned_action.get("description", "").lower()
    )

    state = harness.run("Do the thing")

    assert state.success
    assert len(decisions) == 1
    assert "delete" in decisions[0].planned_action["description"].lower()


def test_checkpoint_and_resume(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    harness = AgentHarness(
        make_llm(), max_iterations=5, checkpoint_dir=str(checkpoint_dir)
    )
    events = collect_events(harness)

    state = harness.run("Do the thing")

    assert EventKind.CHECKPOINT_SAVED in kinds(events)
    path = state.metadata["checkpoint_path"]

    restored = load_checkpoint(path)
    assert restored.goal == state.goal
    assert [t.id for t in restored.tasks] == [t.id for t in state.tasks]
    assert restored.completed

    # Resuming a completed state finishes immediately without re-planning
    harness2 = AgentHarness(make_llm(), max_iterations=5)
    resumed = harness2.run("ignored", initial_state=restored)
    assert resumed.completed


def test_resume_merges_fresh_context():
    state = AgentState(goal="g", tasks=[], metadata={"stale": 1, "keep": "old"})
    state.completed = True

    harness = AgentHarness(make_llm(), max_iterations=2)
    resumed = harness.run("ignored", context={"keep": "new"}, initial_state=state)

    assert resumed.metadata["keep"] == "new"
    assert resumed.metadata["stale"] == 1


def test_failed_dependency_fails_dependents_and_completes_run():
    llm = make_llm(
        plan_tasks=[
            {"id": "task_1", "description": "Base", "dependencies": []},
            {"id": "task_2", "description": "Needs base", "dependencies": ["task_1"]},
            {"id": "task_3", "description": "Needs 2", "dependencies": ["task_2"]},
        ]
    )
    llm.get_completion_with_tools = Mock(side_effect=Exception("boom"))

    harness = AgentHarness(llm, max_iterations=10)
    state = harness.run("Do the thing")

    # The run must terminate decisively, not dangle with completed=False
    assert state.completed and not state.success
    assert state.error
    dependent = state.get_task_by_id("task_2")
    transitive = state.get_task_by_id("task_3")
    assert dependent.status == TaskStatus.FAILED
    assert "task_1" in (dependent.error or "")
    assert transitive.status == TaskStatus.FAILED


def test_resume_grants_fresh_iteration_budget():
    state = AgentState(
        goal="g",
        tasks=[Task.create("finish me")],
        iteration=3,
        max_iterations=3,
        completed=True,
        success=False,
        error="Maximum iterations reached",
    )

    harness = AgentHarness(make_llm(), max_iterations=3)
    resumed = harness.run("ignored", initial_state=state)

    assert resumed.iteration > 3  # actually ran
    assert resumed.completed and resumed.success


def test_state_roundtrip():
    state = AgentState(goal="g", tasks=[Task.create("do it")], iteration=2)
    state.tasks[0].status = TaskStatus.COMPLETED
    state.add_thought("thinking")

    restored = AgentState.from_dict(state.to_dict())

    assert restored.goal == "g"
    assert restored.iteration == 2
    assert restored.tasks[0].status == TaskStatus.COMPLETED
    assert restored.thought_history == ["thinking"]


def test_skip_decision_still_checkpoints(tmp_path):
    def skip(point):
        return SteeringDecision(action=SteeringAction.SKIP)

    harness = AgentHarness(
        make_llm(
            plan_tasks=[{"id": "task_1", "description": "Only", "dependencies": []}]
        ),
        max_iterations=2,
        approval_policy=ApprovalPolicy.ALWAYS,
        decision_provider=skip,
        checkpoint_dir=str(tmp_path),
    )

    state = harness.run("Do the thing")

    assert state.iteration >= 1
    checkpoints = list(tmp_path.glob("*.json"))
    assert checkpoints, "skipped iterations should still write a checkpoint"


def test_pending_stop_skips_decision_provider():
    calls = []

    def provider(point):
        calls.append(point)
        return SteeringDecision(action=SteeringAction.APPROVE)

    harness = AgentHarness(
        make_llm(),
        max_iterations=5,
        approval_policy=ApprovalPolicy.ALWAYS,
        decision_provider=provider,
    )

    # Request a stop before the approval gate is reached
    def stop_on_thought(event: AgentEvent) -> None:
        if event.kind == EventKind.THOUGHT:
            harness.request_stop()

    harness.add_listener(stop_on_thought)
    state = harness.run("Do the thing")

    assert state.completed and not state.success
    assert calls == []  # provider never consulted once a stop is pending


def test_stop_decision_still_checkpoints(tmp_path):
    def stop(point):
        return SteeringDecision(action=SteeringAction.STOP)

    harness = AgentHarness(
        make_llm(),
        max_iterations=5,
        approval_policy=ApprovalPolicy.ALWAYS,
        decision_provider=stop,
        checkpoint_dir=str(tmp_path),
    )

    state = harness.run("Do the thing")

    assert state.completed and not state.success
    assert list(tmp_path.glob("*.json")), "stop should leave a resumable checkpoint"


def test_modify_feedback_reaches_action_and_is_consumed():
    llm = make_llm(
        plan_tasks=[
            {"id": "task_1", "description": "First", "dependencies": []},
            {"id": "task_2", "description": "Second", "dependencies": []},
        ]
    )
    decisions = iter(
        [
            SteeringDecision(action=SteeringAction.MODIFY, feedback="be careful"),
            SteeringDecision(action=SteeringAction.APPROVE),
        ]
    )

    harness = AgentHarness(
        llm,
        max_iterations=5,
        approval_policy=ApprovalPolicy.ALWAYS,
        decision_provider=lambda point: next(decisions),
    )

    state = harness.run("Do the thing")

    prompts = [c.args[0] for c in llm.get_completion_with_tools.call_args_list]
    assert "Operator guidance (must be followed): be careful" in prompts[0]
    # Guidance applies to the modified action only, then is consumed
    assert "be careful" not in prompts[1]
    assert "user_feedback" not in state.metadata


def test_missing_decision_provider_auto_approves():
    harness = AgentHarness(
        make_llm(),
        max_iterations=5,
        approval_policy=ApprovalPolicy.ALWAYS,
        decision_provider=None,
    )

    state = harness.run("Do the thing")

    assert state.success


def test_empty_plan_fails_cleanly():
    llm = make_llm(plan_tasks=[])
    # Empty tasks list -> planner falls back to a single task from the goal,
    # so instead simulate a planner that raises and returns nothing.
    llm.query = Mock(return_value="not json at all, no braces")

    harness = AgentHarness(llm, max_iterations=3)
    state = harness.run("Vague goal")

    # Planner falls back to a single task built from the goal
    assert state.completed
    assert len(state.tasks) == 1


# ----------------------------------------------------------------------
# parsing helpers
# ----------------------------------------------------------------------
def test_extract_json_block_fenced():
    text = 'Here you go:\n```json\n{"a": 1}\n```\nthanks'
    assert json.loads(extract_json_block(text)) == {"a": 1}


def test_extract_json_block_uppercase_and_odd_labels():
    assert json.loads(extract_json_block('```JSON\n{"a": 1}\n```')) == {"a": 1}
    assert json.loads(extract_json_block('```python\n{"a": 1}\n```')) == {"a": 1}


def test_parse_json_response_falls_back_past_bad_fence():
    text = '```\nnot json\n```\nBut here: {"a": 1} as promised'
    assert parse_json_response(text) == {"a": 1}


def test_extract_json_block_bare_object_with_prose():
    text = 'Sure! {"a": {"b": 2}} hope that helps'
    assert json.loads(extract_json_block(text)) == {"a": {"b": 2}}


def test_parse_json_response_default():
    assert parse_json_response("garbage", default={"x": 1}) == {"x": 1}


def test_parse_json_response_raises_without_default():
    with pytest.raises((ValueError, json.JSONDecodeError)):
        parse_json_response("garbage")


def test_parse_json_response_rejects_non_object():
    assert parse_json_response("[1, 2]", default={"ok": True}) == {"ok": True}
