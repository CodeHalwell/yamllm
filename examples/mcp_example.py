"""
Example script demonstrating MCP support in YAMLLM.

This script uses a configuration with MCP connectors and shows how
to interact with tools from MCP servers.

Requirements:
    - Set OPENAI_API_KEY environment variable
    - Set ZAPIER_API_KEY environment variable (if using Zapier MCP)
    - Set CUSTOM_MCP_KEY environment variable (if using custom MCP server)
"""

import os
import sys
from dotenv import load_dotenv
from rich.console import Console
from yamllm.core.llm import LLM

# Load environment variables
load_dotenv()

# Initialize console
console = Console()

def main():
    """Main function."""
    # Check for required environment variables
    if not os.environ.get("OPENAI_API_KEY"):
        console.print("[bold red]Error: OPENAI_API_KEY environment variable not set[/bold red]")
        sys.exit(1)
    
    # Initialize LLM with MCP configuration
    config_path = os.path.join(os.path.dirname(__file__), "../.config_examples/mcp_config.yaml")
    
    try:
        llm = LLM(config_path=config_path, api_key=os.environ.get("OPENAI_API_KEY"))
        console.print("[bold green]Successfully initialized LLM with MCP support[/bold green]")
        
        # Print available tools (including MCP tools)
        tools = llm._prepare_tools()
        console.print("\n[bold]Available Tools:[/bold]")
        for tool in tools:
            if hasattr(tool, "is_mcp_tool") and tool.is_mcp_tool:
                console.print(f"[bold cyan]MCP Tool:[/bold cyan] {tool.name} - {tool.description}")
            else:
                console.print(f"[bold yellow]Local Tool:[/bold yellow] {tool.name} - {tool.description}")
        
        # Example query using tools
        console.print("\n[bold]Example query:[/bold]")
        query = "What's the weather in New York and can you create a reminder in my calendar?"
        console.print(f"[bold blue]User:[/bold blue] {query}")
        
        response = llm.query(query)
        console.print(f"[bold green]AI:[/bold green] {response}")
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        sys.exit(1)

if __name__ == "__main__":
    main()