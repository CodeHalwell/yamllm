"""Argument-parsing regression tests for the agent CLI."""

import argparse

from yamllm.cli.agent import setup_agent_commands


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
