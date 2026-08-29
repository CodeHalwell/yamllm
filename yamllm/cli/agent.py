"""CLI commands for agent operations."""

import argparse
import json

from rich.console import Console
from rich.text import Text

from yamllm import LLM
from yamllm.agent import Agent, WorkflowManager, SimpleAgent
from yamllm.agent.events import AgentEvent, EventKind
from yamllm.agent.harness import AgentHarness, ApprovalPolicy, load_checkpoint
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
    run_parser.add_argument("--max-wall-time", type=float, help="Wall-clock budget in seconds")
    run_parser.add_argument("--simple", action="store_true", help="Use simple agent (no planning)")
    run_parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Approve each action before it runs (Textual TUI unless --plain)"
    )
    run_parser.add_argument(
        "--tui", action="store_true",
        help="Show the live Textual dashboard even without approval gating"
    )
    run_parser.add_argument(
        "--plain", action="store_true",
        help="Plain console output; with --interactive, prompt-based approvals"
    )
    run_parser.add_argument("--auto-approve", action="store_true", help="Auto-approve all actions (with interactive)")
    run_parser.add_argument("--checkpoint-dir", help="Save a state checkpoint after each iteration")
    run_parser.add_argument("--resume", help="Resume from a checkpoint file")
    run_parser.add_argument("--record", action="store_true", help="Record the session for replay")
    run_parser.add_argument("--recording-dir", help="Directory for session recordings")
    run_parser.add_argument("--events-jsonl", help="Append the run's event stream to a JSONL file")
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


def _console_event_renderer(event: AgentEvent) -> None:
    """Render a harness event as an incremental console line."""
    payload = event.payload

    if event.kind == EventKind.PLAN_CREATED:
        console.print(f"[cyan]Plan created: {len(payload.get('tasks', []))} tasks[/cyan]")
        for task in payload.get("tasks", []):
            console.print(f"  [dim]{task.get('id')}[/dim] {task.get('description')}")
    elif event.kind == EventKind.ITERATION_STARTED:
        console.rule(
            f"[dim]iteration {payload.get('iteration')}/{payload.get('max_iterations')}[/dim]"
        )
    elif event.kind == EventKind.THOUGHT:
        console.print(Text(f"💭 {payload.get('thought', '')}", style="italic dim"))
    elif event.kind == EventKind.ACTION_STARTED:
        task = payload.get("task", {})
        console.print(f"[yellow]⚡ {task.get('description', '')}[/yellow]")
    elif event.kind == EventKind.ACTION_FINISHED:
        result = payload.get("result", {})
        if result.get("success"):
            response = (result.get("response") or "").strip()
            if len(response) > 300:
                response = response[:300] + "…"
            if response:
                console.print(Text(response))
        else:
            console.print(f"[red]✗ {result.get('error', 'failed')}[/red]")
    elif event.kind == EventKind.OBSERVATION:
        learned = payload.get("learned")
        if learned:
            console.print(f"[green]👁 {learned}[/green]")
    elif event.kind == EventKind.BUDGET_EXCEEDED:
        console.print(f"[bold red]⛔ {payload.get('detail', 'budget exceeded')}[/bold red]")
    elif event.kind == EventKind.ERROR:
        console.print(f"[bold red]Error: {payload.get('error')}[/bold red]")


def _build_harness(llm, args: argparse.Namespace, approval_policy: ApprovalPolicy) -> AgentHarness:
    """Create an AgentHarness from CLI arguments."""
    return AgentHarness(
        llm,
        max_iterations=args.max_iterations,
        max_wall_time=getattr(args, "max_wall_time", None),
        approval_policy=approval_policy,
        checkpoint_dir=getattr(args, "checkpoint_dir", None),
        enable_recording=getattr(args, "record", False),
        recording_dir=getattr(args, "recording_dir", None),
    )


def run_agent(args: argparse.Namespace) -> int:
    """Run agent with specified goal."""
    try:
        # Load LLM
        console.print(f"[cyan]Loading LLM from {args.config}...[/cyan]")
        llm = LLM(args.config)

        # Load context if provided
        context = None
        if args.context:
            with open(args.context, 'r') as f:
                context = json.load(f)

        # Simple agent: single task, no planning, no steering
        if args.simple:
            console.print("[yellow]Using SimpleAgent (no planning)[/yellow]")
            agent = SimpleAgent(llm)
            state = agent.execute(args.goal, context)
            return _report_result(state, args)

        # Resume support
        initial_state = None
        if getattr(args, "resume", None):
            initial_state = load_checkpoint(args.resume)
            console.print(
                f"[cyan]Resuming from {args.resume} "
                f"(iteration {initial_state.iteration})[/cyan]"
            )

        interactive = args.interactive and not args.auto_approve
        approval_policy = ApprovalPolicy.ALWAYS if interactive else ApprovalPolicy.NEVER
        harness = _build_harness(llm, args, approval_policy)

        jsonl_file = None
        if getattr(args, "events_jsonl", None):
            jsonl_file = open(args.events_jsonl, "a", buffering=1)
            harness.add_listener(lambda ev: jsonl_file.write(ev.to_json() + "\n"))

        goal = initial_state.goal if initial_state else args.goal

        try:
            use_tui = (args.interactive or args.tui) and not args.plain
            if use_tui:
                try:
                    from yamllm.ui.agent_tui import run_agent_tui
                except ImportError:
                    console.print(
                        "[yellow]Textual is not installed "
                        "(pip install textual); falling back to plain output.[/yellow]"
                    )
                    use_tui = False

            if use_tui:
                state = run_agent_tui(harness, goal, context, initial_state=initial_state)
                if state is None:
                    console.print("[red]Run did not produce a final state[/red]")
                    return 1
            else:
                if interactive:
                    # Legacy prompt-based approvals on the plain console
                    from yamllm.agent.interactive_steering import InteractiveSteering

                    steering = InteractiveSteering(console=console)
                    harness.decision_provider = steering.request_decision
                    console.print(
                        "[bold cyan]Interactive mode:[/bold cyan] "
                        "[dim]you will review each action before it runs.[/dim]"
                    )

                harness.add_listener(_console_event_renderer)
                console.print("\n[bold green]Starting agent execution...[/bold green]\n")
                console.print(f"[bold]Goal:[/bold] {goal}\n")
                state = harness.run(goal, context, initial_state=initial_state)
        finally:
            if jsonl_file:
                jsonl_file.close()

        return _report_result(state, args)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1


def _report_result(state, args: argparse.Namespace) -> int:
    """Render the completion summary and persist output if requested."""
    console.print("\n")
    ui = AgentUI(console)
    ui.render_completion(state)

    if getattr(state, "metadata", {}).get("checkpoint_path"):
        console.print(f"[dim]Checkpoint: {state.metadata['checkpoint_path']}[/dim]")
    if getattr(state, "metadata", {}).get("recording_path"):
        console.print(f"[dim]Recording: {state.metadata['recording_path']}[/dim]")

    if args.output:
        save_agent_result(state, args.output)
        console.print(f"\n[green]Result saved to {args.output}[/green]")

    return 0 if state.success else 1


def run_workflow(args: argparse.Namespace) -> int:
    """Run predefined workflow."""
    try:
        # Load LLM
        llm = LLM(args.config)

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
            console.print("[bold]Steps:[/bold]")
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
        llm = LLM(args.config)
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
        llm = LLM(args.config)
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
