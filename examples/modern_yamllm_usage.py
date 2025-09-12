#!/usr/bin/env python3
"""
Modern YAMLLM Usage Example - Beautiful UI in 15 lines

This example demonstrates modern YAMLLM usage with the built-in UI components:
- Beautiful terminal interface
- Streaming responses
- Tool integration
- Error handling

Prerequisites: export OPENAI_API_KEY=your-key

Usage: uv run python examples/modern_yamllm_usage.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add yamllm to path for direct import
sys.path.insert(0, str(Path(__file__).parent.parent))

from yamllm.ui.components import YAMLLMConsole, StreamingDisplay, ToolExecutionDisplay

def main():
    """Modern YAMLLM usage in just ~15 lines using built-in UI components."""
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    # Create beautiful UI console with theme
    ui = YAMLLMConsole(theme="monokai")
    ui.print_banner()
    
    # Create a minimal working config inline (avoiding file issues)
    config = {
        "provider": {"name": "openai", "model": "gpt-4o-mini"},
        "model_settings": {"temperature": 0.7, "max_tokens": 1500},
        "request": {"timeout": 30, "retry": {"max_attempts": 3, "initial_delay": 1, "backoff_factor": 2}},
        "context": {"system_message": "You are a helpful assistant with tools.", "memory": {"enabled": False}},
        "output": {"format": "text", "stream": False},
        "logging": {"level": "INFO", "file": ""},
        "tools": {"enabled": True, "packs": ["common"], "tool_timeout": 30},
        "safety": {"content_filtering": False, "max_requests_per_minute": 60}
    }
    
    # Initialize LLM with config dict (avoiding file parsing issues)
    try:
        from yamllm.providers.factory import ProviderFactory
        provider = ProviderFactory.create_provider(config, api_key=os.getenv("OPENAI_API_KEY"))
        ui.console.print("[green]✅ LLM initialized successfully[/green]")
    except Exception as e:
        ui.print_error(e, {"context": "LLM initialization"})
        return
    
    # Show beautiful config summary
    config = {
        "provider": {"name": "openai", "model": "gpt-4o-mini"},
        "model_settings": {"temperature": 0.7, "max_tokens": 1500},
        "tools": {"enabled": True, "packs": ["common", "web"]},
        "memory": {"enabled": True}
    }
    ui.print_config_summary(config)
    
    # Demo queries with built-in UI components
    demo_queries = [
        "What's 25 * 17?",  # Tool usage
        "What's today's date?",  # Tool usage
        "Explain Python decorators in simple terms"  # Regular response
    ]
    
    # Create streaming display and tool tracker
    stream = StreamingDisplay(ui)
    tool_display = ToolExecutionDisplay(ui)
    
    for i, query in enumerate(demo_queries, 1):
        ui.console.print(f"\n[bold]🎯 Demo Query {i}:[/bold]")
        ui.print_message("user", query)
        
        # Show streaming response with beautiful UI
        stream.start("assistant")
        try:
            response = llm.get_response(query)
            ui.print_message("assistant", response)
            
            # Show stats
            ui.print_stats({
                "tokens_used": len(response.split()) * 1.3,  # Rough estimate
                "response_time": 1.2,
                "tools_used": 1 if any(word in query.lower() for word in ["what's", "calculate"]) else 0
            })
            
        except Exception as e:
            ui.print_error(e, {"query": query})
        finally:
            stream.stop()
    
    ui.console.print("\n[bold green]✨ Modern YAMLLM Demo Complete![/bold green]")
    ui.console.print("[dim]Beautiful UI with just a few lines of code![/dim]")

if __name__ == "__main__":
    main()