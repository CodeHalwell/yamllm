[← Back to Index](index.md)

# Examples

YAMLLM provides comprehensive examples showcasing both the enhanced CLI features and modern API usage patterns.

## 🚀 Quick Start

**New to YAMLLM?** Use the interactive setup:

```bash
# Interactive setup wizard
yamllm init

# Try CLI features showcase
uv run python examples/cli_showcase.py

# Run modern usage example  
uv run python examples/modern_yamllm_usage.py
```

## 📁 Example Files

### 🎯 Feature Showcase Examples

- **`examples/cli_showcase.py`** - Demonstrates all CLI commands
- **`examples/enhanced_chat_demo.py`** - Enhanced chat with slash commands  
- **`examples/config_management_demo.py`** - Configuration templates
- **`examples/modern_yamllm_usage.py`** - Complete modern API patterns

### 🏢 Provider Examples

Each provider has a modernized example showing unified API usage:

- **`examples/openai/openai_example.py`** - OpenAI/GPT models
- **`examples/anthropic/anthropic_example.py`** - Anthropic/Claude models
- **`examples/google/google_example.py`** - Google/Gemini models
- **`examples/mistral/mistral_example.py`** - Mistral models
- **`examples/azure_openai/azure_openai_example.py`** - Azure OpenAI
- **`examples/deepseek/deepseek_example.py`** - DeepSeek models
- **`examples/openrouter/openrouter_example.py`** - OpenRouter models

## 🎯 Modern API Usage

### Unified LLM Interface
```python
from yamllm.core.llm import LLM

# Simple usage
llm = LLM(config_path="my_config.yaml")
response = llm.get_response("Hello!")

# Streaming usage
for chunk in llm.get_streaming_response("Explain AI"):
    print(chunk, end='', flush=True)
```

### Enhanced Error Handling
```python
from yamllm.core.exceptions import ProviderError, ToolExecutionError

try:
    response = llm.get_response("Calculate 2+2")
except ProviderError as e:
    print(f"Provider error: {e}")
except ToolExecutionError as e:
    print(f"Tool error: {e.message}")
```

## 🎨 Enhanced CLI Features

### Interactive Setup
```bash
yamllm init                    # Setup wizard for new users
yamllm status                  # System health check  
yamllm quickstart             # Quick start guide
```

### Configuration Management
```bash
yamllm config create --provider openai --preset casual
yamllm config validate my_config.yaml
yamllm config presets         # List available presets
yamllm config models           # List available models
```

### Tool Management
```bash
yamllm tools list             # Show available tools
yamllm tools manage           # Interactive tool management
yamllm tools test calculator  # Test a specific tool
yamllm tools info weather     # Get tool information
```

### Theme System
```bash
yamllm theme list             # Show available themes
yamllm theme set synthwave    # Apply a theme
yamllm theme preview dark     # Preview a theme
yamllm theme current          # Show current theme
```

### Enhanced Chat
```bash
yamllm chat --config my_config.yaml --enhanced
```

In enhanced chat, use slash commands:
- `/help` - Show available commands
- `/history` - View conversation history  
- `/save session_name` - Save current conversation
- `/multiline` - Enable multiline input
- `/theme synthwave` - Change theme

## 📊 Configuration Presets

The template system includes optimized presets:

- **`casual`** - General conversation with basic tools
- **`coding`** - Development tasks with file and text tools
- **`research`** - Research tasks with web tools and extended memory  
- **`minimal`** - Lightweight setup with minimal tools

Create configs easily:
```bash
yamllm config create --provider openai --preset coding --output my_config.yaml
```

## 🔧 Configuration Examples

### Modern Configuration (Generated)
```yaml
# Generated with: yamllm config create --provider openai --preset casual

provider:
  name: "openai"
  api_key: "${OPENAI_API_KEY}"
  base_url: null

model_settings:
  name: "gpt-4o-mini"
  temperature: 0.7
  max_tokens: 1500
  stream: true

context:
  system_message: |
    You are a helpful AI assistant with access to various tools.
    Be conversational and friendly.

tools:
  enabled: true
  packs: ["common", "web"]
  timeout: 30

memory:
  enabled: true
  max_turns: 20
  store_path: "./memory/sessions"
```

### Tool Integration Example
```python
from yamllm.core.llm import LLM

# Create LLM with tools enabled
llm = LLM(config_path="config_with_tools.yaml")

# Ask questions that use tools
response = llm.get_response("What's 15 * 23?")  # Uses calculator
response = llm.get_response("What's the weather in Tokyo?")  # Uses weather tool
response = llm.get_response("Search for Python async best practices")  # Uses web search
```

## 🛠️ Environment Setup

### Quick Setup
```bash
# 1. Install dependencies (if using source)
uv sync

# 2. Set API key
export OPENAI_API_KEY="your-key"

# 3. Run interactive setup
yamllm init
```

### Manual Setup
```bash
# Set environment variables
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"  
export GOOGLE_API_KEY="your-key"

# Create configuration
yamllm config create --provider openai --preset casual

# Test the setup
yamllm status
```

## 🎯 Running Examples

All examples use `uv` for dependency management:

```bash
# Feature showcase
uv run python examples/cli_showcase.py

# Provider examples
uv run python examples/openai/openai_example.py
uv run python examples/anthropic/anthropic_example.py

# Advanced features
uv run python examples/modern_yamllm_usage.py
uv run python examples/enhanced_chat_demo.py
```

## 📚 Additional Resources

- **[Configuration Guide](configuration.md)** - Complete configuration reference
- **[Tools Documentation](tools.md)** - Available tools and tool packs  
- **[Architecture Guide](architecture.md)** - System architecture overview
- **[CLAUDE.md](../CLAUDE.md)** - Development commands and guidelines
