"""
Tool management CLI commands for YAMLLM.

This module contains all tool-related CLI commands extracted from the
monolithic cli.py for better separation of concerns.
"""

import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


def list_tools(args: argparse.Namespace) -> int:
    """List available tools and tool packs."""
    from yamllm.core.tool_management import tool_manager
    
    console.print("\n[bold cyan]Available Tools[/bold cyan]\n")
    
    # If pack specified, show pack details
    if hasattr(args, 'pack') and args.pack:
        pack_tools = tool_manager.get_pack_tools(args.pack)
        if not pack_tools:
            console.print(f"[red]Unknown pack: {args.pack}[/red]")
            return 1
        
        console.print(f"[bold]Pack:[/bold] {args.pack}")
        console.print(f"[bold]Tools:[/bold] {', '.join(pack_tools)}\n")
        
        table = Table(box=box.ROUNDED)
        table.add_column("Tool", style="cyan")
        table.add_column("Description", style="white")
        
        for tool_name in pack_tools:
            info = tool_manager.get_tool_info(tool_name)
            if info:
                table.add_row(tool_name, info.get('description', 'No description'))
        
        console.print(table)
        return 0
    
    # Show all packs
    console.print("[bold]Tool Packs:[/bold]")
    packs = tool_manager.get_all_packs()
    
    pack_table = Table(box=box.ROUNDED)
    pack_table.add_column("Pack", style="cyan")
    pack_table.add_column("Tools", style="white")
    pack_table.add_column("Description", style="white")
    
    for pack_name, pack_info in packs.items():
        tools_str = ', '.join(pack_info.get('tools', []))[:50]
        if len(tools_str) >= 50:
            tools_str += '...'
        pack_table.add_row(
            pack_name,
            tools_str,
            pack_info.get('description', 'No description')
        )
    
    console.print(pack_table)
    
    # Show all individual tools
    console.print("\n[bold]Individual Tools:[/bold]")
    all_tools = tool_manager.get_all_tools()
    
    # Filter by category if specified
    if hasattr(args, 'category') and args.category:
        all_tools = {k: v for k, v in all_tools.items() 
                    if v.get('category') == args.category}
    
    tools_table = Table(box=box.ROUNDED)
    tools_table.add_column("Tool", style="cyan")
    tools_table.add_column("Category", style="yellow")
    tools_table.add_column("Description", style="white")
    
    for tool_name, tool_info in sorted(all_tools.items()):
        tools_table.add_row(
            tool_name,
            tool_info.get('category', 'general'),
            tool_info.get('description', 'No description')
        )
    
    console.print(tools_table)
    console.print(f"\n[dim]Use 'yamllm tools info <tool>' for detailed information[/dim]")
    
    return 0


def show_tool_info(args: argparse.Namespace) -> int:
    """Show detailed information about a specific tool."""
    from yamllm.core.tool_management import tool_manager
    
    tool_name = args.tool
    info = tool_manager.get_tool_info(tool_name)
    
    if not info:
        console.print(f"[red]Tool not found: {tool_name}[/red]")
        return 1
    
    console.print(f"\n[bold cyan]Tool: {tool_name}[/bold cyan]\n")
    console.print(f"[bold]Description:[/bold] {info.get('description', 'No description')}")
    console.print(f"[bold]Category:[/bold] {info.get('category', 'general')}")
    
    # Show parameters
    params = info.get('parameters', {})
    if params:
        console.print("\n[bold]Parameters:[/bold]")
        param_table = Table(box=box.ROUNDED)
        param_table.add_column("Parameter", style="cyan")
        param_table.add_column("Type", style="yellow")
        param_table.add_column("Required", style="magenta")
        param_table.add_column("Description", style="white")
        
        properties = params.get('properties', {})
        required = params.get('required', [])
        
        for param_name, param_info in properties.items():
            param_table.add_row(
                param_name,
                param_info.get('type', 'any'),
                "Yes" if param_name in required else "No",
                param_info.get('description', 'No description')
            )
        
        console.print(param_table)
    
    # Show example
    example = info.get('example')
    if example:
        console.print(f"\n[bold]Example:[/bold]")
        console.print(Panel(str(example), border_style="dim"))
    
    return 0


def test_tool(args: argparse.Namespace) -> int:
    """Test a tool interactively."""
    console.print(f"\n[bold cyan]Testing tool: {args.tool}[/bold cyan]\n")
    console.print("[yellow]Tool testing is not yet implemented[/yellow]")
    console.print("[dim]This will allow interactive testing of tool functionality[/dim]")
    return 0


def manage_tools(args: argparse.Namespace) -> int:
    """Interactive tool management interface."""
    console.print("\n[bold cyan]Tool Management[/bold cyan]\n")
    console.print("[yellow]Interactive tool management is not yet implemented[/yellow]")
    console.print("[dim]This will allow enabling/disabling tools and managing configurations[/dim]")
    return 0


def search_tools(args: argparse.Namespace) -> int:
    """Search for tools by query."""
    from yamllm.core.tool_management import tool_manager
    
    query = args.query.lower()
    all_tools = tool_manager.get_all_tools()
    
    # Search in tool names and descriptions
    results = {}
    for tool_name, tool_info in all_tools.items():
        if (query in tool_name.lower() or 
            query in tool_info.get('description', '').lower()):
            results[tool_name] = tool_info
    
    if not results:
        console.print(f"[yellow]No tools found matching '{args.query}'[/yellow]")
        return 0
    
    console.print(f"\n[bold cyan]Search Results for '{args.query}':[/bold cyan]\n")
    
    table = Table(box=box.ROUNDED)
    table.add_column("Tool", style="cyan")
    table.add_column("Description", style="white")
    
    for tool_name, tool_info in sorted(results.items()):
        table.add_row(
            tool_name,
            tool_info.get('description', 'No description')
        )
    
    console.print(table)
    console.print(f"\n[dim]Found {len(results)} tool(s)[/dim]")
    
    return 0


def setup_tools_commands(subparsers):
    """Set up tools-related CLI commands."""
    tools_cmd = subparsers.add_parser("tools", help="Tool management and information")
    tools_sub = tools_cmd.add_subparsers(dest="tools_action", help="Tools actions")
    
    # Tools list (default)
    tools_list_cmd = tools_sub.add_parser("list", help="List available tools and tool packs")
    tools_list_cmd.add_argument("--pack", help="Show details for a specific tool pack")
    tools_list_cmd.add_argument("--category", help="Filter by category")
    tools_list_cmd.set_defaults(func=list_tools)
    
    # Tools info
    tools_info_cmd = tools_sub.add_parser("info", help="Show detailed information about a tool")
    tools_info_cmd.add_argument("tool", help="Tool name")
    tools_info_cmd.set_defaults(func=show_tool_info)
    
    # Tools test
    tools_test_cmd = tools_sub.add_parser("test", help="Test a tool")
    tools_test_cmd.add_argument("tool", help="Tool name")
    tools_test_cmd.set_defaults(func=test_tool)
    
    # Tools manage
    tools_manage_cmd = tools_sub.add_parser("manage", help="Interactive tool management")
    tools_manage_cmd.set_defaults(func=manage_tools)
    
    # Tools search
    tools_search_cmd = tools_sub.add_parser("search", help="Search tools")
    tools_search_cmd.add_argument("query", help="Search query")
    tools_search_cmd.set_defaults(func=search_tools)
    
    # Default tools action (for backward compatibility)
    tools_cmd.set_defaults(func=list_tools)
