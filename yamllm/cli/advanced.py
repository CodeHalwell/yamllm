"""CLI commands for advanced features (ensemble, cost tracking, routing, replay)."""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from yamllm import LLM
from yamllm.core.ensemble import EnsembleManager, EnsembleStrategy, ParallelEnsembleManager
from yamllm.core.model_router import ModelRouter
from yamllm.agent.recording import SessionPlayer, RecordingManager

console = Console()


def setup_advanced_commands(subparsers):
    """Setup advanced CLI commands."""

    # yamllm ensemble
    ensemble_parser = subparsers.add_parser(
        "ensemble",
        help="Multi-model ensemble execution"
    )
    ensemble_subparsers = ensemble_parser.add_subparsers(dest="ensemble_command", help="Ensemble commands")

    # yamllm ensemble run
    run_parser = ensemble_subparsers.add_parser(
        "run",
        help="Run prompt across multiple models"
    )
    run_parser.add_argument("prompt", help="Prompt to execute")
    run_parser.add_argument("--config", required=True, action="append", help="Config files for each provider (can specify multiple)")
    run_parser.add_argument("--strategy", choices=["consensus", "best_of_n", "voting", "first_success"],
                           default="consensus", help="Ensemble strategy")
    run_parser.add_argument("--timeout", type=float, default=30.0, help="Timeout per model")
    run_parser.add_argument("--async", dest="use_async", action="store_true", help="Use async execution")
    run_parser.add_argument("--output", "-o", help="Save result to file")
    run_parser.set_defaults(func=run_ensemble)

    # yamllm cost
    cost_parser = subparsers.add_parser(
        "cost",
        help="Cost tracking and optimization"
    )
    cost_subparsers = cost_parser.add_subparsers(dest="cost_command", help="Cost commands")

    # yamllm cost report
    report_parser = cost_subparsers.add_parser(
        "report",
        help="Show cost report for current session"
    )
    report_parser.add_argument("--config", required=True, help="Config file path")
    report_parser.set_defaults(func=show_cost_report)

    # yamllm cost optimize
    optimize_parser = cost_subparsers.add_parser(
        "optimize",
        help="Get cost optimization suggestions"
    )
    optimize_parser.add_argument("--config", required=True, help="Config file path")
    optimize_parser.set_defaults(func=optimize_costs)

    # yamllm route
    route_parser = subparsers.add_parser(
        "route",
        help="Intelligent model routing"
    )
    route_parser.add_argument("prompt", help="Prompt to analyze")
    route_parser.add_argument("--explain", action="store_true", help="Explain routing decision")
    route_parser.set_defaults(func=route_model)

    # yamllm replay
    replay_parser = subparsers.add_parser(
        "replay",
        help="Replay agent sessions"
    )
    replay_subparsers = replay_parser.add_subparsers(dest="replay_command", help="Replay commands")

    # yamllm replay session
    session_parser = replay_subparsers.add_parser(
        "session",
        help="Replay a session recording"
    )
    session_parser.add_argument("recording", help="Path to recording file")
    session_parser.add_argument("--speed", type=float, default=1.0, help="Playback speed")
    session_parser.add_argument("--until", type=int, help="Stop at iteration N")
    session_parser.set_defaults(func=replay_session)

    # yamllm replay list
    list_parser = replay_subparsers.add_parser(
        "list",
        help="List available recordings"
    )
    list_parser.add_argument("--dir", default="./recordings", help="Recordings directory")
    list_parser.set_defaults(func=list_recordings)

    # yamllm replay compare
    compare_parser = replay_subparsers.add_parser(
        "compare",
        help="Compare two session recordings"
    )
    compare_parser.add_argument("recording1", help="First recording file")
    compare_parser.add_argument("recording2", help="Second recording file")
    compare_parser.set_defaults(func=compare_recordings)

    return ensemble_parser


def run_ensemble(args: argparse.Namespace) -> int:
    """Run ensemble execution across multiple models."""
    try:
        console.print(f"[cyan]Loading {len(args.config)} model configurations...[/cyan]")

        # Load LLMs
        llms = {}
        for i, config_path in enumerate(args.config):
            llm = LLM(config_path=config_path)
            provider_name = f"{llm.provider_name}_{i}" if len(args.config) > 1 else llm.provider_name
            llms[provider_name] = llm

        console.print(f"[green]Loaded {len(llms)} models[/green]")
        console.print(f"[cyan]Strategy: {args.strategy}[/cyan]\n")

        # Create ensemble manager
        if args.use_async:
            import asyncio
            manager = ParallelEnsembleManager(llms)

            async def run():
                return await manager.execute_async(
                    prompt=args.prompt,
                    strategy=EnsembleStrategy(args.strategy),
                    timeout=args.timeout
                )

            result = asyncio.run(run())
        else:
            manager = EnsembleManager(llms)
            result = manager.execute(
                prompt=args.prompt,
                strategy=EnsembleStrategy(args.strategy),
                timeout=args.timeout
            )

        # Display results
        console.print(Panel.fit(
            f"[bold]Final Response[/bold]\n\n{result.final_response}",
            title="Ensemble Result",
            border_style="green"
        ))

        console.print(f"\n[bold]Agreement Score:[/bold] {result.agreement_score:.2%}")
        if result.selected_model:
            console.print(f"[bold]Selected Model:[/bold] {result.selected_model}")
        console.print(f"[bold]Reasoning:[/bold] {result.reasoning}\n")

        # Show individual responses
        console.print("[bold cyan]Individual Model Responses:[/bold cyan]\n")
        for resp in result.responses:
            if resp.error:
                console.print(f"[red]{resp.provider}/{resp.model}: Error - {resp.error}[/red]")
            else:
                console.print(f"[green]{resp.provider}/{resp.model}[/green] ({resp.execution_time:.2f}s)")
                console.print(f"  {resp.response[:200]}...\n")

        # Save output if requested
        if args.output:
            output_data = {
                "strategy": result.strategy.value,
                "final_response": result.final_response,
                "agreement_score": result.agreement_score,
                "selected_model": result.selected_model,
                "reasoning": result.reasoning,
                "responses": [
                    {
                        "provider": r.provider,
                        "model": r.model,
                        "response": r.response,
                        "execution_time": r.execution_time,
                        "error": r.error
                    }
                    for r in result.responses
                ]
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            console.print(f"[green]Result saved to {args.output}[/green]")

        return 0

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1


def show_cost_report(args: argparse.Namespace) -> int:
    """Show cost report for current session."""
    try:
        llm = LLM(config_path=args.config)
        summary = llm.get_cost_summary()

        # Create cost report table
        table = Table(title="Cost Report", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Total Cost", f"${summary.total_cost:.4f}")
        table.add_row("Total Calls", str(summary.total_calls))
        table.add_row("Total Tokens", f"{summary.total_tokens:,}")
        table.add_row("Avg Cost/Call", f"${summary.avg_cost_per_call:.4f}")

        if summary.budget_limit:
            usage_pct = (summary.total_cost / summary.budget_limit) * 100
            table.add_row("Budget Limit", f"${summary.budget_limit:.2f}")
            table.add_row("Budget Used", f"{usage_pct:.1f}%")

        console.print(table)

        # Show provider breakdown
        if summary.provider_breakdown:
            console.print("\n[bold cyan]Cost by Provider:[/bold cyan]\n")
            for provider, cost in summary.provider_breakdown.items():
                console.print(f"  {provider}: ${cost:.4f}")

        return 0

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1


def optimize_costs(args: argparse.Namespace) -> int:
    """Get cost optimization suggestions."""
    try:
        llm = LLM(config_path=args.config)
        suggestions = llm.get_cost_optimization_suggestions()

        console.print("\n[bold cyan]Cost Optimization Suggestions:[/bold cyan]\n")

        if suggestions.get("current_model"):
            console.print(f"[bold]Current Model:[/bold] {suggestions['current_model']}")
            console.print(f"[bold]Current Cost:[/bold] ${suggestions.get('total_cost', 0):.4f}\n")

        if suggestions.get("recommendations"):
            console.print("[bold green]Recommendations:[/bold green]\n")
            for rec in suggestions["recommendations"]:
                console.print(f"  • Switch to {rec['model']}")
                console.print(f"    - Cost: ${rec['estimated_cost']:.4f}")
                console.print(f"    - Savings: ${rec['savings']:.4f} ({rec['savings_percent']:.1f}%)")
                console.print(f"    - Use for: {rec['use_case']}\n")

        return 0

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1


def route_model(args: argparse.Namespace) -> int:
    """Analyze prompt and suggest optimal model."""
    try:
        router = ModelRouter()
        provider, model, reasoning = router.select_model(args.prompt)

        console.print(Panel.fit(
            f"[bold cyan]Recommended Model:[/bold cyan]\n\n"
            f"[bold]Provider:[/bold] {provider}\n"
            f"[bold]Model:[/bold] {model}\n\n"
            f"[bold]Reasoning:[/bold]\n{reasoning}",
            title="Model Routing",
            border_style="green"
        ))

        if args.explain:
            task_type, complexity = router.analyze_task(args.prompt)
            console.print(f"\n[bold]Task Analysis:[/bold]")
            console.print(f"  Type: {task_type.value}")
            console.print(f"  Complexity: {complexity.value}")

        return 0

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1


def replay_session(args: argparse.Namespace) -> int:
    """Replay a session recording."""
    try:
        player = SessionPlayer(args.recording)

        console.print(f"[cyan]Replaying session: {player.session_id}[/cyan]")
        console.print(f"[bold]Goal:[/bold] {player.goal}\n")

        player.replay(speed=args.speed, until_iteration=args.until)

        return 0

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1


def list_recordings(args: argparse.Namespace) -> int:
    """List available session recordings."""
    try:
        manager = RecordingManager(args.dir)
        recordings = manager.list_recordings()

        if not recordings:
            console.print(f"[yellow]No recordings found in {args.dir}[/yellow]")
            return 0

        table = Table(title=f"Session Recordings ({args.dir})", show_header=True, header_style="bold cyan")
        table.add_column("Session ID", style="cyan")
        table.add_column("Goal", style="white")
        table.add_column("Iterations", style="yellow")
        table.add_column("Date", style="green")

        for rec in recordings:
            table.add_row(
                rec["session_id"],
                rec["goal"][:50] + "..." if len(rec["goal"]) > 50 else rec["goal"],
                str(rec["iterations"]),
                rec["timestamp"]
            )

        console.print(table)
        return 0

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1


def compare_recordings(args: argparse.Namespace) -> int:
    """Compare two session recordings."""
    try:
        player1 = SessionPlayer(args.recording1)
        player2 = SessionPlayer(args.recording2)

        comparison = player1.compare_with(player2)

        console.print("\n[bold cyan]Session Comparison:[/bold cyan]\n")

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="cyan")
        table.add_column("Session 1", style="white")
        table.add_column("Session 2", style="white")

        table.add_row("Session ID", comparison["session1"]["session_id"], comparison["session2"]["session_id"])
        table.add_row("Goal", comparison["session1"]["goal"][:30], comparison["session2"]["goal"][:30])
        table.add_row("Iterations", str(comparison["session1"]["iterations"]), str(comparison["session2"]["iterations"]))
        table.add_row("Timestamp", comparison["session1"]["timestamp"], comparison["session2"]["timestamp"])

        console.print(table)

        if comparison["differences"]:
            console.print("\n[bold yellow]Key Differences:[/bold yellow]\n")
            for diff in comparison["differences"]:
                console.print(f"  • {diff}")

        return 0

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1
