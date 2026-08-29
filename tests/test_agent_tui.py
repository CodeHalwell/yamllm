"""Headless tests for the Textual agent TUI."""

import asyncio
import json

import pytest
from unittest.mock import Mock

pytest.importorskip("textual")

from yamllm.agent.harness import AgentHarness, ApprovalPolicy  # noqa: E402
from yamllm.ui.agent_tui import AgentTUI, ApprovalScreen  # noqa: E402


def make_llm(plan_tasks=None):
    plan = {
        "tasks": (
            plan_tasks
            if plan_tasks is not None
            else [
                {"id": "task_1", "description": "Inspect the repo", "dependencies": []},
                {
                    "id": "task_2",
                    "description": "Write the fix",
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


async def wait_for(pilot, predicate, ticks=100, delay=0.05):
    for _ in range(ticks):
        await pilot.pause(delay)
        if predicate():
            return True
    return False


def test_tui_dashboard_completes_run():
    async def scenario():
        harness = AgentHarness(
            make_llm(), max_iterations=5, approval_policy=ApprovalPolicy.NEVER
        )
        app = AgentTUI(harness, "Fix the bug")
        async with app.run_test(size=(120, 40)) as pilot:
            finished = await wait_for(pilot, lambda: app.final_state is not None)
            assert finished, "agent run did not finish"
            assert app.final_state.completed and app.final_state.success
            table = app.query_one("#tasks")
            assert table.row_count == 2
            await pilot.press("q")

    asyncio.run(scenario())


def test_tui_approval_modal_approve():
    async def scenario():
        harness = AgentHarness(
            make_llm(
                plan_tasks=[{"id": "task_1", "description": "Only", "dependencies": []}]
            ),
            max_iterations=3,
            approval_policy=ApprovalPolicy.ALWAYS,
        )
        app = AgentTUI(harness, "Goal")
        async with app.run_test(size=(120, 40)) as pilot:
            shown = await wait_for(
                pilot, lambda: isinstance(app.screen, ApprovalScreen)
            )
            assert shown, "approval modal never appeared"
            await pilot.press("a")
            finished = await wait_for(pilot, lambda: app.final_state is not None)
            assert finished, "run did not finish after approval"
            assert app.final_state.success
            assert [d.action.value for d in harness.decision_history] == ["approve"]
            await pilot.press("q")

    asyncio.run(scenario())


def test_tui_stop_binding_cancels_run():
    async def scenario():
        harness = AgentHarness(
            make_llm(), max_iterations=3, approval_policy=ApprovalPolicy.ALWAYS
        )
        app = AgentTUI(harness, "Goal")
        async with app.run_test(size=(120, 40)) as pilot:
            shown = await wait_for(
                pilot, lambda: isinstance(app.screen, ApprovalScreen)
            )
            assert shown
            # Stop from the approval modal
            await pilot.press("x")
            finished = await wait_for(pilot, lambda: app.final_state is not None)
            assert finished
            assert not app.final_state.success
            await pilot.press("q")

    asyncio.run(scenario())
