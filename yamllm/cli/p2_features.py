"""CLI commands for P2 features: Multi-Agent and Learning System."""

import click
import json

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box


console = Console()


# Helper functions
def _get_llm(config: str = None, model: str = "gpt-4"):
    """
    Get LLM instance from config or create default.

    Args:
        config: Optional config file path
        model: Default model to use if no config provided (deprecated, use config file)

    Returns:
        LLM instance
    
    Raises:
        ValueError: If no config file is provided or found
    """
    from yamllm import LLM
    import os

    if config and os.path.exists(config):
        return LLM(config)
    else:
        # Try default config locations
        default_configs = [
            "config/openai.yaml",
            ".config_examples/openai_example.yaml",
            os.path.expanduser("~/.yamllm/config.yaml")
        ]
        for default_config in default_configs:
            if os.path.exists(default_config):
                return LLM(default_config)
        
        raise ValueError(
            "No config file provided or found. Please specify a config file path "
            "or create one at config/openai.yaml"
        )


@click.group(name="multi-agent")
def multi_agent_group():
    """Multi-agent collaboration commands."""
    pass


@multi_agent_group.command(name="execute")
@click.argument("goal")
@click.option("--config", "-c", type=click.Path(exists=True), help="LLM config file")
@click.option("--roles", "-r", multiple=True, help="Agent roles to use (e.g., coder, reviewer)")
@click.option("--max-iterations", "-m", default=10, help="Maximum coordination iterations")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def multi_agent_execute(goal: str, config: str, roles: tuple, max_iterations: int, verbose: bool):
    """
    Execute a goal using collaborative multi-agent system.

    Example:
        yamllm multi-agent execute "Build a REST API" --roles coder --roles reviewer --roles tester
    """
    try:
        from yamllm.agent.multi_agent import (
            AgentCoordinator, CollaborativeAgent, AgentCapability, AgentRole
        )

        console.print(f"[bold cyan]Multi-Agent Collaborative Execution[/bold cyan]\n")
        console.print(f"[yellow]Goal:[/yellow] {goal}\n")

        # Load LLM
        llm = _get_llm(config)
        if not config:
            console.print("[yellow]No config provided, using default OpenAI GPT-4[/yellow]")

        # Create coordinator
        coordinator = AgentCoordinator(coordinator_llm=llm)

        # Parse and register agents with specified roles
        if roles:
            role_map = {
                "coordinator": AgentRole.COORDINATOR,
                "researcher": AgentRole.RESEARCHER,
                "coder": AgentRole.CODER,
                "reviewer": AgentRole.REVIEWER,
                "tester": AgentRole.TESTER,
                "debugger": AgentRole.DEBUGGER,
                "documenter": AgentRole.DOCUMENTER,
                "analyst": AgentRole.ANALYST,
            }

            for i, role_name in enumerate(roles):
                role = role_map.get(role_name.lower())
                if not role:
                    console.print(f"[red]Unknown role: {role_name}[/red]")
                    continue

                # Create agent
                agent = CollaborativeAgent(
                    agent_id=f"{role_name}_{i}",
                    llm=llm,
                    capability=AgentCapability(
                        role=role,
                        skills=[role_name],
                        max_concurrent_tasks=2
                    )
                )
                coordinator.register_agent(agent)
        else:
            # Default: Create standard agent set
            default_roles = [
                (AgentRole.RESEARCHER, ["research", "gather_info"]),
                (AgentRole.CODER, ["code", "implement"]),
                (AgentRole.REVIEWER, ["review", "validate"]),
            ]

            for role, skills in default_roles:
                agent = CollaborativeAgent(
                    agent_id=f"{role.value}_agent",
                    llm=llm,
                    capability=AgentCapability(
                        role=role,
                        skills=skills,
                        max_concurrent_tasks=2
                    )
                )
                coordinator.register_agent(agent)

        # Show registered agents
        status = coordinator.get_status()
        console.print(f"[green]Registered {status['registered_agents']} agents[/green]\n")

        if verbose:
            for agent_id, agent_info in status["agents"].items():
                console.print(f"  • {agent_id} ({agent_info['role']})")

        # Execute collaborative task
        with console.status("[bold green]Executing collaborative task...[/bold green]"):
            result = coordinator.execute_collaborative_task(goal, max_iterations=max_iterations)

        # Display results
        console.print(f"\n[bold cyan]═══ Results ═══[/bold cyan]\n")
        console.print(f"[green]✓ Completed {result['tasks_completed']} tasks in {result['iterations']} iterations[/green]\n")

        # Show task results
        if verbose and result.get("results"):
            for task_id, task_result in result["results"].items():
                console.print(Panel(
                    str(task_result),
                    title=f"Task: {task_id}",
                    border_style="blue"
                ))

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise click.Abort()


@multi_agent_group.command(name="status")
@click.option("--config", "-c", type=click.Path(exists=True), help="LLM config file")
def multi_agent_status(config: str):
    """Show multi-agent system status."""
    console.print("[cyan]Multi-Agent System Status[/cyan]\n")
    console.print("Use 'multi-agent execute' to start a collaborative task.")


@click.group(name="learn")
def learning_group():
    """Agent learning and improvement commands."""
    pass


@learning_group.command(name="record")
@click.argument("task_description")
@click.option("--outcome", "-o", type=click.Choice(["success", "failure", "partial", "timeout", "error"]), required=True)
@click.option("--duration", "-d", type=float, required=True, help="Task duration in seconds")
@click.option("--actions", "-a", type=str, help="JSON string of actions taken")
@click.option("--details", type=str, help="JSON string of outcome details")
@click.option("--db", type=click.Path(), default="agent_learning.db", help="Learning database path")
def record_experience(task_description: str, outcome: str, duration: float, actions: str, details: str, db: str):
    """
    Record an agent experience for learning.

    Example:
        yamllm learn record "Fix bug in API" --outcome success --duration 120.5 --actions '[{"action": "analyze"}]'
    """
    try:
        from yamllm.agent.learning_system import LearningSystem, OutcomeType

        # Parse actions
        actions_list = json.loads(actions) if actions else []
        details_dict = json.loads(details) if details else {}

        # Create learning system (requires LLM for analysis)
        llm = _get_llm()
        learning = LearningSystem(llm, storage_path=db)

        # Record experience
        outcome_type = OutcomeType(outcome)
        experience = learning.record_experience(
            task_description=task_description,
            actions=actions_list,
            outcome=outcome_type,
            outcome_details=details_dict,
            duration=duration
        )

        console.print(f"[green]✓ Recorded experience: {experience.experience_id}[/green]")
        console.print(f"  Task: {task_description}")
        console.print(f"  Outcome: {outcome}")
        console.print(f"  Duration: {duration:.1f}s")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise click.Abort()


@learning_group.command(name="analyze")
@click.option("--config", "-c", type=click.Path(exists=True), help="LLM config file")
@click.option("--db", type=click.Path(), default="agent_learning.db", help="Learning database path")
@click.option("--min-experiences", "-m", default=10, help="Minimum experiences needed")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed analysis")
def analyze_learning(config: str, db: str, min_experiences: int, verbose: bool):
    """
    Analyze experiences and generate learning insights.

    Example:
        yamllm learn analyze --min-experiences 5 --verbose
    """
    try:
        from yamllm.agent.learning_system import LearningSystem

        console.print("[bold cyan]Learning Analysis[/bold cyan]\n")

        # Load LLM
        llm = _get_llm(config)

        # Create learning system
        learning = LearningSystem(llm, storage_path=db)

        # Analyze
        with console.status("[bold green]Analyzing experiences...[/bold green]"):
            insights = learning.analyze_and_learn(min_experiences=min_experiences)

        if not insights:
            console.print("[yellow]Not enough experiences to generate insights[/yellow]")
            return

        console.print(f"[green]✓ Generated {len(insights)} new insights[/green]\n")

        # Display insights
        table = Table(title="Learning Insights", box=box.ROUNDED)
        table.add_column("#", style="cyan", width=4)
        table.add_column("Type", style="yellow")
        table.add_column("Pattern", style="white")
        table.add_column("Confidence", style="green")

        for i, insight in enumerate(insights[:10], 1):
            table.add_row(
                str(i),
                insight.improvement_type.value,
                insight.pattern[:50] + "..." if len(insight.pattern) > 50 else insight.pattern,
                f"{insight.confidence:.1%}"
            )

        console.print(table)

        if verbose:
            console.print("\n[bold]Recommendations:[/bold]\n")
            for i, insight in enumerate(insights[:5], 1):
                console.print(f"{i}. {insight.recommendation}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise click.Abort()


@learning_group.command(name="recommend")
@click.argument("task_description")
@click.option("--config", "-c", type=click.Path(exists=True), help="LLM config file")
@click.option("--db", type=click.Path(), default="agent_learning.db", help="Learning database path")
def get_recommendations(task_description: str, config: str, db: str):
    """
    Get recommendations for a task based on learning.

    Example:
        yamllm learn recommend "Implement authentication feature"
    """
    try:
        from yamllm.agent.learning_system import LearningSystem

        console.print(f"[bold cyan]Recommendations for:[/bold cyan] {task_description}\n")

        # Load LLM
        llm = _get_llm(config)

        # Create learning system
        learning = LearningSystem(llm, storage_path=db)

        # Get recommendations
        recommendations = learning.get_recommendations(task_description)

        if not recommendations:
            console.print("[yellow]No recommendations available yet. Record more experiences![/yellow]")
            return

        console.print(f"[green]Found {len(recommendations)} recommendations:[/green]\n")

        for i, rec in enumerate(recommendations, 1):
            console.print(f"[bold]{i}.[/bold] {rec}\n")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise click.Abort()


@learning_group.command(name="metrics")
@click.option("--db", type=click.Path(), default="agent_learning.db", help="Learning database path")
@click.option("--export", "-e", type=click.Path(), help="Export metrics to JSON file")
def show_metrics(db: str, export: str):
    """
    Show learning performance metrics.

    Example:
        yamllm learn metrics --export metrics.json
    """
    try:
        from yamllm.agent.learning_system import LearningSystem

        # Create learning system
        llm = _get_llm()
        learning = LearningSystem(llm, storage_path=db)

        # Get metrics
        metrics = learning.get_metrics()
        summary = learning.get_learning_summary()

        # Display metrics
        console.print("[bold cyan]Learning Metrics[/bold cyan]\n")

        info_table = Table(box=box.SIMPLE)
        info_table.add_column("Metric", style="cyan")
        info_table.add_column("Value", style="white")

        info_table.add_row("Total Experiences", str(metrics.total_tasks))
        info_table.add_row("Successful Tasks", str(metrics.successful_tasks))
        info_table.add_row("Failed Tasks", str(metrics.failed_tasks))
        info_table.add_row("Success Rate", f"{metrics.success_rate:.1%}")
        info_table.add_row("Average Duration", f"{metrics.average_duration:.1f}s")
        info_table.add_row("Total Insights", str(summary["total_insights"]))

        console.print(info_table)

        # Show insights by type
        if summary.get("insights_by_type"):
            console.print("\n[bold]Insights by Type:[/bold]\n")
            insights_table = Table(box=box.SIMPLE)
            insights_table.add_column("Type", style="yellow")
            insights_table.add_column("Count", style="green")

            for insight_type, count in summary["insights_by_type"].items():
                insights_table.add_row(insight_type, str(count))

            console.print(insights_table)

        # Export if requested
        if export:
            with open(export, 'w') as f:
                json.dump({
                    "metrics": {
                        "total_tasks": metrics.total_tasks,
                        "successful_tasks": metrics.successful_tasks,
                        "failed_tasks": metrics.failed_tasks,
                        "success_rate": metrics.success_rate,
                        "average_duration": metrics.average_duration
                    },
                    "summary": summary
                }, f, indent=2)
            console.print(f"\n[green]✓ Metrics exported to {export}[/green]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise click.Abort()


@learning_group.command(name="export")
@click.argument("output_path", type=click.Path())
@click.option("--config", "-c", type=click.Path(exists=True), help="LLM config file")
@click.option("--db", type=click.Path(), default="agent_learning.db", help="Learning database path")
def export_knowledge(output_path: str, config: str, db: str):
    """
    Export learned knowledge to a file.

    Example:
        yamllm learn export knowledge.json
    """
    try:
        from yamllm.agent.learning_system import LearningSystem

        # Load LLM
        llm = _get_llm(config)

        # Create learning system
        learning = LearningSystem(llm, storage_path=db)

        # Export
        learning.export_knowledge(output_path)

        console.print(f"[green]✓ Knowledge exported to {output_path}[/green]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise click.Abort()


@learning_group.command(name="import")
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--config", "-c", type=click.Path(exists=True), help="LLM config file")
@click.option("--db", type=click.Path(), default="agent_learning.db", help="Learning database path")
def import_knowledge(input_path: str, config: str, db: str):
    """
    Import learned knowledge from a file.

    Example:
        yamllm learn import knowledge.json
    """
    try:
        from yamllm.agent.learning_system import LearningSystem

        # Load LLM
        llm = _get_llm(config)

        # Create learning system
        learning = LearningSystem(llm, storage_path=db)

        # Import
        learning.import_knowledge(input_path)

        console.print(f"[green]✓ Knowledge imported from {input_path}[/green]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise click.Abort()


# Export command groups
def register_p2_commands(cli):
    """Register P2 commands with main CLI."""
    cli.add_command(multi_agent_group)
    cli.add_command(learning_group)
