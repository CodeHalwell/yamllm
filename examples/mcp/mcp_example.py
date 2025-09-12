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
import random
from dotenv import load_dotenv
from yamllm.core.llm import LLM
from yamllm.ui.components import YAMLLMConsole, StreamingDisplay

# Load environment variables
load_dotenv()

def main():
    """Main function."""
    # Check for required environment variables
    if not os.environ.get("OPENAI_API_KEY"):
        console.print("[bold red]Error: OPENAI_API_KEY environment variable not set[/bold red]")
        sys.exit(1)
    
    # Initialize LLM with MCP configuration
    config_path = os.path.join(os.path.dirname(__file__), "../.config_examples/mcp/mcp_config.yaml")
    
    try:
        llm = LLM(config_path=config_path, api_key=os.environ.get("OPENAI_API_KEY"))
        ui = YAMLLMConsole(theme=random.choice(["default", "monokai", "dracula"]))
        stream = StreamingDisplay(ui)
        ui.print_banner()
        ui.console.print("MCP chat — type 'exit' to quit.\n", style="dim")

        while True:
            prompt = ui.prompt_user("You")
            if not prompt or prompt.strip().lower() == "exit":
                break
            ui.print_message("user", prompt)
            stream.start("assistant")
            llm.set_stream_callback(stream.update)
            try:
                llm.query(prompt)
            finally:
                stream.stop()
                llm.clear_stream_callback()
        # Print available tools (including MCP tools)
        tools = llm._prepare_tools()
        console.print("\n[bold]Available Tools:[/bold]")
        for tool in tools:
            # tool is a provider-friendly dict; display function name/description
            fn = tool.get("function", {}) if isinstance(tool, dict) else {}
            console.print(f"- {fn.get('name', 'unknown')} — {fn.get('description', '')}")
        
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
