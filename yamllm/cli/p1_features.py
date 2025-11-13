"""CLI commands for P1 features (dynamic tools, code intelligence, advanced git)."""

import argparse
import sys
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

from yamllm import LLM

console = Console()


def setup_p1_commands(subparsers):
    """Setup P1 feature CLI commands."""

    # yamllm tool
    tool_parser = subparsers.add_parser(
        "tool",
        help="Dynamic tool creation and management"
    )
    tool_subparsers = tool_parser.add_subparsers(dest="tool_command", help="Tool commands")

    # yamllm tool create
    create_parser = tool_subparsers.add_parser(
        "create",
        help="Create tool from natural language description"
    )
    create_parser.add_argument("description", help="Tool description")
    create_parser.add_argument("--name", help="Tool name (auto-generated if not provided)")
    create_parser.add_argument("--config", required=True, help="LLM config file")
    create_parser.add_argument("--export", help="Export tool to file")
    create_parser.set_defaults(func=create_dynamic_tool)

    # yamllm code
    code_parser = subparsers.add_parser(
        "code",
        help="Code context intelligence and analysis"
    )
    code_subparsers = code_parser.add_subparsers(dest="code_command", help="Code commands")

    # yamllm code analyze
    analyze_parser = code_subparsers.add_parser(
        "analyze",
        help="Analyze code repository"
    )
    analyze_parser.add_argument("path", help="Repository path")
    analyze_parser.add_argument("--query", help="Optional query for relevant context")
    analyze_parser.add_argument("--output", "-o", help="Save analysis to file")
    analyze_parser.add_argument("--config", help="LLM config for summaries")
    analyze_parser.set_defaults(func=analyze_code)

    # yamllm code context
    context_parser = code_subparsers.add_parser(
        "context",
        help="Extract relevant code context for query"
    )
    context_parser.add_argument("path", help="Repository path")
    context_parser.add_argument("query", help="Query or task description")
    context_parser.add_argument("--max-symbols", type=int, default=10, help="Max symbols to include")
    context_parser.add_argument("--config", help="LLM config")
    context_parser.set_defaults(func=get_code_context)

    # yamllm git
    git_parser = subparsers.add_parser(
        "git",
        help="Advanced git workflow automation"
    )
    git_subparsers = git_parser.add_subparsers(dest="git_command", help="Git commands")

    # yamllm git smart-commit
    smart_commit_parser = git_subparsers.add_parser(
        "smart-commit",
        help="Create commit with AI-generated message"
    )
    smart_commit_parser.add_argument("--repo", default=".", help="Repository path")
    smart_commit_parser.add_argument("--files", nargs="+", help="Specific files to commit")
    smart_commit_parser.add_argument("--message", help="Override auto-generated message")
    smart_commit_parser.add_argument("--config", help="LLM config")
    smart_commit_parser.set_defaults(func=smart_commit)

    # yamllm git smart-branch
    smart_branch_parser = git_subparsers.add_parser(
        "smart-branch",
        help="Create intelligently named branch"
    )
    smart_branch_parser.add_argument("task", help="Task description")
    smart_branch_parser.add_argument("--repo", default=".", help="Repository path")
    smart_branch_parser.add_argument("--strategy", choices=["gitflow", "github_flow", "trunk_based"],
                                     default="github_flow", help="Branching strategy")
    smart_branch_parser.set_defaults(func=smart_branch)

    # yamllm git auto-pr
    auto_pr_parser = git_subparsers.add_parser(
        "auto-pr",
        help="Generate PR title and description"
    )
    auto_pr_parser.add_argument("--repo", default=".", help="Repository path")
    auto_pr_parser.add_argument("--base", default="main", help="Base branch")
    auto_pr_parser.add_argument("--config", help="LLM config")
    auto_pr_parser.set_defaults(func=auto_pr)

    # yamllm git status
    git_status_parser = git_subparsers.add_parser(
        "status",
        help="Enhanced git status"
    )
    git_status_parser.add_argument("--repo", default=".", help="Repository path")
    git_status_parser.set_defaults(func=git_status)

    return tool_parser


def create_dynamic_tool(args: argparse.Namespace) -> int:
    """Create a dynamic tool from natural language."""
    try:
        console.print(f"[cyan]Creating tool from description...[/cyan]")
        console.print(f"[dim]{args.description}[/dim]\n")

        # Load LLM
        with LLM(config_path=args.config) as llm:
            tool = llm.create_dynamic_tool(args.description, name=args.name)

        console.print(Panel.fit(
            f"[bold green]Tool Created Successfully![/bold green]\n\n"
            f"[bold]Name:[/bold] {tool.name}\n"
            f"[bold]Description:[/bold] {tool.description}\n"
            f"[bold]Parameters:[/bold] {json.dumps(tool.parameters, indent=2)}",
            title="Dynamic Tool",
            border_style="green"
        ))

        # Show example
        if tool.example_usage:
            console.print(f"\n[bold]Example Usage:[/bold]\n{tool.example_usage}")

        # Export if requested
        if args.export:
            from yamllm.tools.dynamic_tool_creator import ToolCreator
            creator = ToolCreator(llm)
            creator.created_tools[tool.name] = tool
            creator.export_tool(tool.name, args.export)
            console.print(f"\n[green]Exported to: {args.export}[/green]")

        return 0

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1


def analyze_code(args: argparse.Namespace) -> int:
    """Analyze code repository."""
    try:
        console.print(f"[cyan]Analyzing repository: {args.path}[/cyan]\n")

        from yamllm.code.context_intelligence import CodeContextIntelligence

        # Initialize intelligence
        llm = None
        if args.config:
            llm = LLM(config_path=args.config)

        intel = CodeContextIntelligence(llm)
        context = intel.analyze_project(args.path)

        # Display summary
        console.print(Panel.fit(
            f"[bold]Repository Analysis[/bold]\n\n"
            f"Files: {len(context.files)}\n"
            f"Symbols: {len(context.symbol_index)}\n"
            f"Entry Points: {len(context.entry_points)}",
            title="Summary",
            border_style="cyan"
        ))

        # Show top files by complexity
        top_files = sorted(
            context.files.items(),
            key=lambda x: x[1].complexity_score,
            reverse=True
        )[:5]

        if top_files:
            console.print("\n[bold cyan]Most Complex Files:[/bold cyan]\n")
            table = Table()
            table.add_column("File", style="cyan")
            table.add_column("Lines", style="yellow")
            table.add_column("Symbols", style="green")
            table.add_column("Complexity", style="red")

            for path, ctx in top_files:
                filename = Path(path).name
                table.add_row(
                    filename,
                    str(ctx.lines_of_code),
                    str(len(ctx.symbols)),
                    str(ctx.complexity_score)
                )

            console.print(table)

        # Show architecture summary if available
        if context.architecture_summary:
            console.print(f"\n[bold cyan]Architecture Summary:[/bold cyan]\n")
            console.print(context.architecture_summary)

        # Save if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({
                    "files": len(context.files),
                    "symbols": len(context.symbol_index),
                    "entry_points": context.entry_points,
                    "architecture": context.architecture_summary
                }, f, indent=2)
            console.print(f"\n[green]Saved to: {args.output}[/green]")

        return 0

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1


def get_code_context(args: argparse.Namespace) -> int:
    """Get relevant code context for query."""
    try:
        console.print(f"[cyan]Extracting context for: {args.query}[/cyan]\n")

        from yamllm.code.context_intelligence import CodeContextIntelligence

        llm = None
        if args.config:
            llm = LLM(config_path=args.config)

        intel = CodeContextIntelligence(llm)
        intel.analyze_project(args.path)

        context_text = intel.get_relevant_context(
            args.query,
            max_symbols=args.max_symbols
        )

        console.print(Panel(
            context_text,
            title="Relevant Context",
            border_style="cyan"
        ))

        return 0

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1


def smart_commit(args: argparse.Namespace) -> int:
    """Create smart commit with AI-generated message."""
    try:
        from yamllm.tools.advanced_git import AdvancedGitWorkflow

        llm = None
        if args.config:
            llm = LLM(config_path=args.config)

        git = AdvancedGitWorkflow(args.repo, llm)

        console.print("[cyan]Analyzing changes...[/cyan]\n")

        # Analyze changes
        analysis = git.analyze_changes()

        # Show analysis
        console.print(f"Files changed: {analysis.files_changed}")
        console.print(f"Lines added: [green]+{analysis.lines_added}[/green]")
        console.print(f"Lines deleted: [red]-{analysis.lines_deleted}[/red]")
        console.print(f"Components: {', '.join(analysis.affected_components)}")

        if analysis.breaking_changes:
            console.print("[bold red]⚠ Breaking changes detected![/bold red]")

        console.print(f"\n[bold]Suggested message:[/bold]\n{analysis.suggested_message}\n")

        # Create commit
        commit_hash = git.smart_commit(
            files=args.files,
            message=args.message or analysis.suggested_message
        )

        console.print(f"[green]✓ Commit created: {commit_hash[:8]}[/green]")

        return 0

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1


def smart_branch(args: argparse.Namespace) -> int:
    """Create smart branch."""
    try:
        from yamllm.tools.advanced_git import AdvancedGitWorkflow, BranchStrategy

        git = AdvancedGitWorkflow(args.repo)

        strategy_map = {
            "gitflow": BranchStrategy.GITFLOW,
            "github_flow": BranchStrategy.GITHUB_FLOW,
            "trunk_based": BranchStrategy.TRUNK_BASED
        }

        branch_name = git.smart_branch(
            args.task,
            strategy=strategy_map[args.strategy]
        )

        console.print(f"[green]✓ Created and switched to branch: {branch_name}[/green]")

        return 0

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1


def auto_pr(args: argparse.Namespace) -> int:
    """Generate PR info."""
    try:
        from yamllm.tools.advanced_git import AdvancedGitWorkflow

        llm = None
        if args.config:
            llm = LLM(config_path=args.config)

        git = AdvancedGitWorkflow(args.repo, llm)

        pr_info = git.auto_pr(base=args.base)

        console.print(Panel.fit(
            f"[bold]Title:[/bold] {pr_info['title']}\n\n"
            f"[bold]Base:[/bold] {pr_info['base']}\n"
            f"[bold]Head:[/bold] {pr_info['head']}\n\n"
            f"[bold]Description:[/bold]\n{pr_info['body']}",
            title="Pull Request",
            border_style="cyan"
        ))

        console.print("\n[dim]Use gh CLI to create PR:[/dim]")
        console.print(f"[dim]gh pr create --title \"{pr_info['title']}\" --body \"...\"[/dim]")

        return 0

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1


def git_status(args: argparse.Namespace) -> int:
    """Show enhanced git status."""
    try:
        from yamllm.tools.advanced_git import AdvancedGitWorkflow

        git = AdvancedGitWorkflow(args.repo)
        status = git.get_status()

        # Display status
        console.print(f"\n[bold cyan]Branch:[/bold cyan] {status.branch}")

        if status.ahead > 0 or status.behind > 0:
            console.print(f"[yellow]↑{status.ahead} ↓{status.behind}[/yellow]")

        console.print()

        if status.staged_files:
            console.print("[bold green]Staged files:[/bold green]")
            for f in status.staged_files:
                console.print(f"  [green]✓[/green] {f}")

        if status.unstaged_files:
            console.print("\n[bold yellow]Unstaged changes:[/bold yellow]")
            for f in status.unstaged_files:
                console.print(f"  [yellow]○[/yellow] {f}")

        if status.untracked_files:
            console.print("\n[bold red]Untracked files:[/bold red]")
            for f in status.untracked_files:
                console.print(f"  [red]?[/red] {f}")

        if status.stashed > 0:
            console.print(f"\n[dim]Stashed: {status.stashed}[/dim]")

        if not status.is_dirty:
            console.print("[green]✓ Working tree clean[/green]")

        return 0

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1
