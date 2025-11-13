"""CLI commands for agent operations."""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console

from yamllm import LLM
from yamllm.agent import Agent, WorkflowManager, SimpleAgent
from yamllm.ui.agent_ui import AgentUI

console = Console()


def setup_agent_commands(subparsers):
    """Setup agent-related CLI commands."""

    # yamllm agent
    agent_parser = subparsers.add_parser(
        "agent",
        help="Autonomous agent operations"
    )
    agent_subparsers = agent_parser.add_subparsers(dest="agent_command", help="Agent commands")

    # yamllm agent run
    run_parser = agent_subparsers.add_parser(
        "run",
        help="Run agent with a goal"
    )
    run_parser.add_argument("goal", help="Goal to achieve")
    run_parser.add_argument("--config", required=True, help="Config file path")
    run_parser.add_argument("--context", help="JSON context file")
    run_parser.add_argument("--max-iterations", type=int, default=10, help="Max iterations")
    run_parser.add_argument("--simple", action="store_true", help="Use simple agent (no planning)")
    run_parser.add_argument("--output", "-o", help="Save result to file")
    run_parser.set_defaults(func=run_agent)

    # yamllm agent workflow
    workflow_parser = agent_subparsers.add_parser(
        "workflow",
        help="Run predefined workflow"
    )
    workflow_parser.add_argument("workflow", help="Workflow name")
    workflow_parser.add_argument("--config", required=True, help="Config file path")
    workflow_parser.add_argument("--context", help="JSON context (required fields depend on workflow)")
    workflow_parser.add_argument("--list", action="store_true", help="List available workflows")
    workflow_parser.add_argument("--info", action="store_true", help="Show workflow info")
    workflow_parser.add_argument("--output", "-o", help="Save result to file")
    workflow_parser.set_defaults(func=run_workflow)

    # yamllm agent debug
    debug_parser = agent_subparsers.add_parser(
        "debug",
        help="Debug a bug (shortcut for debug workflow)"
    )
    debug_parser.add_argument("description", help="Bug description")
    debug_parser.add_argument("--config", required=True, help="Config file path")
    debug_parser.add_argument("--file", help="File path where bug occurs")
    debug_parser.add_argument("--error", help="Error message")
    debug_parser.add_argument("--output", "-o", help="Save result to file")
    debug_parser.set_defaults(func=debug_bug)

    # yamllm agent implement
    implement_parser = agent_subparsers.add_parser(
        "implement",
        help="Implement a feature (shortcut for implement workflow)"
    )
    implement_parser.add_argument("description", help="Feature description")
    implement_parser.add_argument("--config", required=True, help="Config file path")
    implement_parser.add_argument("--requirements", help="Additional requirements")
    implement_parser.add_argument("--output", "-o", help="Save result to file")
    implement_parser.set_defaults(func=implement_feature)

    return agent_parser


def run_agent(args: argparse.Namespace) -> int:
    """Run agent with specified goal."""
    try:
        # Load LLM
        console.print(f"[cyan]Loading LLM from {args.config}...[/cyan]")
        llm = LLM(config_path=args.config)

        # Load context if provided
        context = None
        if args.context:
            with open(args.context, 'r') as f:
                context = json.load(f)

        # Create agent
        if args.simple:
            agent = SimpleAgent(llm)
            console.print("[yellow]Using SimpleAgent (no planning)[/yellow]")
        else:
            agent = Agent(llm, max_iterations=args.max_iterations)
            console.print(f"[cyan]Using full Agent (max {args.max_iterations} iterations)[/cyan]")

        # Create UI
        ui = AgentUI(console)

        # Setup progress callback
        def progress_callback(state):
            ui.render_full_state(state)

        agent.progress_callback = progress_callback

        # Execute
        console.print(f"\n[bold green]Starting agent execution...[/bold green]\n")
        console.print(f"[bold]Goal:[/bold] {args.goal}\n")

        state = agent.execute(args.goal, context)

        # Show completion
        console.print("\n")
        ui.render_completion(state)

        # Save output if requested
        if args.output:
            save_agent_result(state, args.output)
            console.print(f"\n[green]Result saved to {args.output}[/green]")

        return 0 if state.success else 1

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1


def run_workflow(args: argparse.Namespace) -> int:
    """Run predefined workflow."""
    try:
        # Load LLM
        llm = LLM(config_path=args.config)

        # Create agent and workflow manager
        agent = Agent(llm)
        manager = WorkflowManager(agent)

        # Handle list
        if args.list:
            workflows = manager.list_workflows()
            console.print("\n[bold cyan]Available Workflows:[/bold cyan]\n")

            for wf in workflows:
                console.print(f"[bold]{wf['name']}[/bold]")
                console.print(f"  {wf['description']}")
                console.print(f"  Required: {', '.join(wf['required_context'])}")
                if wf['optional_context']:
                    console.print(f"  Optional: {', '.join(wf['optional_context'])}")
                console.print()

            return 0

        # Handle info
        if args.info:
            info = manager.get_workflow_info(args.workflow)
            console.print(f"\n[bold cyan]{info['name']}[/bold cyan]\n")
            console.print(f"[bold]Description:[/bold] {info['description']}\n")
            console.print(f"[bold]Steps:[/bold]")
            for i, step in enumerate(info['steps'], 1):
                console.print(f"  {i}. {step}")
            console.print()
            return 0

        # Execute workflow
        context = json.loads(args.context) if args.context else {}

        console.print(f"\n[bold green]Running workflow: {args.workflow}[/bold green]\n")

        state = manager.execute_workflow(args.workflow, context)

        # Show completion
        ui = AgentUI(console)
        ui.render_completion(state)

        # Save output if requested
        if args.output:
            save_agent_result(state, args.output)

        return 0 if state.success else 1

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1


def debug_bug(args: argparse.Namespace) -> int:
    """Debug a bug using debug workflow."""
    try:
        llm = LLM(config_path=args.config)
        agent = Agent(llm)
        manager = WorkflowManager(agent)

        # Build context
        context = {"bug_description": args.description}
        if args.file:
            context["file_path"] = args.file
        if args.error:
            context["error_message"] = args.error

        console.print(f"\n[bold green]Debugging: {args.description}[/bold green]\n")

        state = manager.execute_workflow("debug_bug", context)

        # Show completion
        ui = AgentUI(console)
        ui.render_completion(state)

        if args.output:
            save_agent_result(state, args.output)

        return 0 if state.success else 1

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1


def implement_feature(args: argparse.Namespace) -> int:
    """Implement a feature using implement workflow."""
    try:
        llm = LLM(config_path=args.config)
        agent = Agent(llm)
        manager = WorkflowManager(agent)

        # Build context
        context = {"feature_description": args.description}
        if args.requirements:
            context["requirements"] = args.requirements

        console.print(f"\n[bold green]Implementing: {args.description}[/bold green]\n")

        state = manager.execute_workflow("implement_feature", context)

        # Show completion
        ui = AgentUI(console)
        ui.render_completion(state)

        if args.output:
            save_agent_result(state, args.output)

        return 0 if state.success else 1

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1


def save_agent_result(state, output_path: str) -> None:
    """Save agent result to file."""
    result = {
        "goal": state.goal,
        "success": state.success,
        "completed": state.completed,
        "iterations": state.iteration,
        "error": state.error,
        "tasks": [
            {
                "id": t.id,
                "description": t.description,
                "status": t.status.value,
                "result": t.result,
                "error": t.error
            }
            for t in state.tasks
        ],
        "learnings": state.metadata.get("learnings", [])
    }

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
