#!/usr/bin/env python3
"""
Enhanced Chat Interface Demo

This example demonstrates the new enhanced chat features including:
- Slash commands (/help, /history, /save, /load, etc.)
- Multiline input support
- Session management
- Theme integration
- Tool integration

Prerequisites:
1. Set your API key: export OPENAI_API_KEY=your-key
2. Create a config: yamllm config create --provider openai --preset casual -o chat_config.yaml

Usage: uv run python examples/enhanced_chat_demo.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def create_demo_config():
    """Create a demo configuration file."""
    config_content = """# Enhanced Chat Demo Configuration
provider:
  name: "openai"
  api_key: "${OPENAI_API_KEY}"
  base_url: null

model_settings:
  name: "gpt-4o-mini"
  temperature: 0.7
  max_tokens: 1500
  top_p: 1.0
  frequency_penalty: 0.0
  presence_penalty: 0.0

request:
  timeout: 30
  max_retries: 3
  retry_delay: 1.0
  stream: true

context:
  system_message: |
    You are a helpful AI assistant. You can use various tools to help answer questions.
    Be conversational and friendly. When using tools, explain what you're doing.

output:
  format: "text"
  show_tokens: false
  show_timing: true

logging:
  level: "INFO"
  file: null

tools:
  enabled: true
  packs: ["common", "web"]
  individual_tools: []
  timeout: 30

memory:
  enabled: true
  max_turns: 20
  store_path: "./memory/chat_demo"
  embedding_provider: null

safety:
  content_filter: false
  max_requests_per_minute: 60
  blocked_phrases: []
"""
    
    config_path = Path("examples/chat_demo_config.yaml")
    config_path.write_text(config_content)
    return config_path

def main():
    """Run the enhanced chat demo."""
    print("🚀 Enhanced Chat Interface Demo")
    print("=" * 50)
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set your OpenAI API key:")
        print("   export OPENAI_API_KEY=your-key")
        sys.exit(1)
    
    # Create demo config
    config_path = create_demo_config()
    print(f"✅ Created demo config: {config_path}")
    
    print("\n🎯 Enhanced Chat Features:")
    print("• /help         - Show available commands")
    print("• /history      - View conversation history") 
    print("• /save <name>  - Save current conversation")
    print("• /load <name>  - Load a saved conversation")
    print("• /multiline    - Enable multiline input mode")
    print("• /clear        - Clear conversation history")
    print("• /theme <name> - Change chat theme")
    print("• /tools        - List available tools")
    print("• /exit         - Exit the chat")
    
    print("\n📋 Try These Demo Interactions:")
    print("1. Ask: 'What's the weather like in New York?'")
    print("2. Use: '/history' to see conversation")
    print("3. Try: '/multiline' for long messages")
    print("4. Use: '/save demo-session' to save")
    print("5. Try: '/theme synthwave' for cool colors")
    
    print("\n🎯 Starting Enhanced Chat...")
    print("=" * 50)
    
    # Start the enhanced chat
    import subprocess
    try:
        subprocess.run([
            "uv", "run", "python", "-m", "yamllm.cli", "chat",
            "--config", str(config_path),
            "--enhanced"
        ])
    except KeyboardInterrupt:
        print("\n👋 Chat demo ended")
    except Exception as e:
        print(f"❌ Error starting chat: {e}")
        print("\nTry running manually:")
        print(f"yamllm chat --config {config_path} --enhanced")

if __name__ == "__main__":
    main()