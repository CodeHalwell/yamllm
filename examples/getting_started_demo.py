#!/usr/bin/env python3
"""
Getting Started with YAMLLM Demo

This example demonstrates the basic workflow for getting started with YAMLLM:
1. Check system status
2. Set up configuration 
3. Basic API patterns
4. CLI integration

Usage: uv run python examples/getting_started_demo.py
"""

import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def run_cli_command(cmd, description, show_output=True):
    """Run a CLI command and optionally display results."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"Command: yamllm {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(
            f"uv run python -m yamllm.cli {cmd}".split(),
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if show_output and result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"Error: {result.stderr}")
        return result.returncode == 0
            
    except subprocess.TimeoutExpired:
        print("⏰ Command timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def demonstrate_configuration():
    """Show configuration management."""
    print("\n📊 Configuration Management Demo")
    print("-" * 40)
    
    # Show presets
    run_cli_command("config presets", "Available Configuration Presets")
    
    # Show models
    run_cli_command("config models", "Available Models")
    
    print("\n💡 To create a configuration:")
    print("yamllm config create --provider openai --preset casual --output my_config.yaml")

def demonstrate_tools():
    """Show tool management."""
    print("\n🔧 Tool Management Demo")
    print("-" * 40)
    
    # List tools
    run_cli_command("tools list", "Available Tools")
    
    # Show web pack
    run_cli_command("tools list --pack common", "Common Tool Pack")

def demonstrate_themes():
    """Show theme system."""
    print("\n🎨 Theme System Demo")
    print("-" * 40)
    
    # List themes
    run_cli_command("theme list", "Available Themes")
    
    # Show current theme
    run_cli_command("theme current", "Current Theme")

def show_basic_api_patterns():
    """Show basic API usage patterns without making real calls."""
    print("\n🎯 Basic API Patterns")
    print("-" * 40)
    
    print("Here are the basic patterns for using YAMLLM:")
    print()
    
    print("📝 Simple Usage:")
    print("```python")
    print("from yamllm.core.llm import LLM")
    print()
    print("llm = LLM(config_path='my_config.yaml', api_key=os.getenv('OPENAI_API_KEY'))")
    print("response = llm.get_response('Hello!')")
    print("print(response)")
    print("```")
    print()
    
    print("🔄 Streaming Usage:")
    print("```python")
    print("for chunk in llm.get_streaming_response('Explain AI'):")
    print("    print(chunk, end='', flush=True)")
    print("```")
    print()
    
    print("⚠️ Error Handling:")
    print("```python")
    print("from yamllm.core.exceptions import ProviderError, ToolExecutionError")
    print()
    print("try:")
    print("    response = llm.get_response('Calculate 2+2')")
    print("except ProviderError as e:")
    print("    print(f'Provider error: {e}')")
    print("except ToolExecutionError as e:")
    print("    print(f'Tool error: {e.message}')")
    print("```")

def main():
    """Main demonstration function."""
    print("🚀 Getting Started with YAMLLM")
    print("=" * 50)
    
    # Check prerequisites
    has_api_key = bool(os.getenv("OPENAI_API_KEY"))
    
    print("📋 Prerequisites Check:")
    print(f"• OpenAI API Key: {'✅ Set' if has_api_key else '❌ Not set'}")
    if not has_api_key:
        print("  Set with: export OPENAI_API_KEY=your-key")
    
    print(f"• Working Directory: {Path.cwd()}")
    print(f"• Python Dependencies: {'✅ Available' if Path('pyproject.toml').exists() else '❌ Missing'}")
    
    # Show system status
    run_cli_command("status", "System Status Check")
    
    # Show providers
    run_cli_command("providers", "Available Providers")
    
    # Configuration demo
    demonstrate_configuration()
    
    # Tools demo
    demonstrate_tools()
    
    # Themes demo
    demonstrate_themes()
    
    # API patterns
    show_basic_api_patterns()
    
    print(f"\n{'='*60}")
    print("🎉 Getting Started Demo Complete!")
    print()
    print("🚀 Next Steps:")
    if not has_api_key:
        print("1. Set your API key: export OPENAI_API_KEY=your-key")
        print("2. Run setup wizard: yamllm init")
    else:
        print("1. Create config: yamllm config create --provider openai --preset casual")
        print("2. Start chatting: yamllm chat --config openai_casual_config.yaml")
    
    print("3. Try enhanced chat: yamllm chat --config my_config.yaml --enhanced")
    print("4. Explore tools: yamllm tools manage")
    print("5. Customize theme: yamllm theme set synthwave")
    print("=" * 60)

if __name__ == "__main__":
    main()