import os
import dotenv
from rich.console import Console
from yamllm.core.llm import AnthropicAI

"""
This script initializes a language model (LLM) using a configuration file and an API key, 
then enters a loop where it takes user input, queries the LLM with the input, and prints the response.
The LLM is configured to use tools like web search and calculator when appropriate.

This example demonstrates the use of Anthropic's Claude model with the traditional LLM interface.
"""

# Initialize console and load environment variables
console = Console()
dotenv.load_dotenv()

# Get the absolute path to the config file
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
config_path = os.path.join(root_dir, ".config_examples", "anthropic_config.yaml")

# Initialize the LLM with config
llm = AnthropicAI(config_path=config_path, api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Display settings including tools configuration
llm.print_settings()

# Verify tool integration
if llm.tools_enabled:
    console.print(f"[green]Tools are enabled![/green] Available tools: {', '.join(llm.tools)}")
    console.print("[yellow]Try asking questions that might need web search or calculations.[/yellow]")
    console.print("Examples:")
    console.print("  - What were the major news headlines today?")
    console.print("  - What is the square root of 1764 divided by 42?")
    console.print("  - Convert 100 kilometers to miles")
else:
    console.print("[red]Tools are not enabled in the configuration.[/red]")


while True:
    try:          
        prompt = input("\nHuman: ")
        if prompt.lower() == "exit":
            break
        
        # The query method should already handle tools through get_response
        response = llm.query(prompt)
        if response is None:
            continue
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting...[/yellow]")
        break
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")

console.print("\n[green]Goodbye![/green]")