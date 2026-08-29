"""Argument-parsing and log-sink regression tests for the agent CLI."""

import argparse
import json
from unittest.mock import Mock, patch

from yamllm.agent.events import AgentEvent, EventKind
from yamllm.cli.agent import _event_for_log, run_agent, setup_agent_commands


def parse(argv):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    setup_agent_commands(sub)
    return parser.parse_args(argv)


def test_run_accepts_output_flag():
    args = parse(["agent", "run", "goal", "--config", "c.yaml", "-o", "out.json"])
    assert args.output == "out.json"


def test_run_output_defaults_to_none():
    args = parse(["agent", "run", "goal", "--config", "c.yaml"])
    assert args.output is None


def test_run_accepts_harness_flags():
    args = parse(
        [
            "agent",
            "run",
            "goal",
            "--config",
            "c.yaml",
            "--max-wall-time",
            "60",
            "--checkpoint-dir",
            ".ckpt",
            "--resume",
            "state.json",
            "--record",
            "--events-jsonl",
            "events.jsonl",
            "--interactive",
            "--auto-approve",
        ]
    )
    assert args.max_wall_time == 60.0
    assert args.checkpoint_dir == ".ckpt"
    assert args.resume == "state.json"
    assert args.record is True
    assert args.events_jsonl == "events.jsonl"
    assert args.interactive is True
    assert args.auto_approve is True


def test_run_accepts_log_thoughts_flag():
    args = parse(["agent", "run", "goal", "--config", "c.yaml", "--log-thoughts"])
    assert args.log_thoughts is True


def test_event_log_redacts_thoughts_by_default():
    event = AgentEvent(kind=EventKind.THOUGHT, payload={"thought": "secret plan"})
    redacted = _event_for_log(event, log_thoughts=False)
    assert redacted.payload["thought"] == "[redacted]"
    # Original event untouched (live UI still sees the thought)
    assert event.payload["thought"] == "secret plan"


def test_event_log_redacts_approval_thought():
    event = AgentEvent(
        kind=EventKind.APPROVAL_REQUESTED,
        payload={"thought": "secret", "planned_action": {"task_id": "t1"}},
    )
    redacted = _event_for_log(event, log_thoughts=False)
    assert redacted.payload["thought"] == "[redacted]"
    assert redacted.payload["planned_action"] == {"task_id": "t1"}


def test_event_log_keeps_thoughts_when_opted_in():
    event = AgentEvent(kind=EventKind.THOUGHT, payload={"thought": "secret plan"})
    kept = _event_for_log(event, log_thoughts=True)
    assert kept.payload["thought"] == "secret plan"


def test_event_log_passes_other_events_through():
    event = AgentEvent(kind=EventKind.RUN_FINISHED, payload={"success": True})
    assert _event_for_log(event, log_thoughts=False) is event


def test_simple_interactive_run_keeps_approval_gating():
    """--simple --interactive must still gate actions (routed via harness)."""
    llm = Mock()
    llm.query = Mock(return_value=json.dumps({}))
    llm.get_completion_with_tools = Mock(
        return_value={"content": "Done", "tool_calls": [], "tool_results": []}
    )

    args = parse(
        [
            "agent",
            "run",
            "do it",
            "--config",
            "c.yaml",
            "--simple",
            "--interactive",
            "--auto-approve",
            "--plain",
        ]
    )

    with patch("yamllm.cli.agent.LLM", return_value=llm):
        rc = run_agent(args)

    assert rc == 0
    # The single seeded task actually executed through the tool-enabled actor
    assert llm.get_completion_with_tools.called
