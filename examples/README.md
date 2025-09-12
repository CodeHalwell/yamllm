# YAMLLM Examples

This directory contains comprehensive examples showcasing YAMLLM's enhanced features and capabilities.

## 🚀 Quick Start

**New to YAMLLM?** Start here:

```bash
# 1. Interactive setup wizard
yamllm init

# 2. Getting started demo (no API calls required)
uv run python examples/getting_started_demo.py

# 3. Try the CLI showcase
uv run python examples/cli_showcase.py

# 4. Run a comprehensive example
uv run python examples/modern_yamllm_usage.py
```

## 📁 Example Structure

### 🎯 Feature Showcase Examples
- **`getting_started_demo.py`** - Beginner-friendly demo (no API calls needed)
- **`cli_showcase.py`** - Demonstrates all CLI commands and features
- **`enhanced_chat_demo.py`** - Enhanced chat interface with slash commands
- **`config_management_demo.py`** - Configuration templates and presets
- **`modern_yamllm_usage.py`** - Complete modern API usage patterns

### 🏢 Provider Examples
Each provider directory contains a modernized example showing:
- Modern LLM class usage (unified interface)
- Enhanced error handling
- Tool integration
- Streaming responses

#### Available Provider Examples:
- **`openai/openai_example.py`** - OpenAI/GPT models
- **`anthropic/anthropic_example.py`** - Anthropic/Claude models  
- **`google/google_example.py`** - Google/Gemini models
- **`mistral/mistral_example.py`** - Mistral models
- **`azure_openai/azure_openai_example.py`** - Azure OpenAI
- **`deepseek/deepseek_example.py`** - DeepSeek models
- **`openrouter/openrouter_example.py`** - OpenRouter models

### 🔧 Advanced Examples
- **`misc/async_example.py`** - Async/await usage patterns
- **`misc/vector_store.py`** - Vector storage and semantic search
- **`mcp/mcp_example.py`** - Model Context Protocol integration

## 🎨 New CLI Features

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
```

### Tool Management
```bash
yamllm tools list             # Show available tools
yamllm tools manage           # Interactive tool management
yamllm tools test calculator  # Test a specific tool
```

### Theme System
```bash
yamllm theme list             # Show available themes
yamllm theme set synthwave    # Apply a theme
yamllm theme preview dark     # Preview a theme
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

The new template system includes optimized presets:

- **`casual`** - General conversation with basic tools
- **`coding`** - Development tasks with file and text tools  
- **`research`** - Research tasks with web tools and extended memory
- **`minimal`** - Lightweight setup with minimal tools

## 🛠️ Prerequisites

### API Keys
Set the appropriate environment variable for your provider:

```bash
export OPENAI_API_KEY="your-key"        # OpenAI
export ANTHROPIC_API_KEY="your-key"     # Anthropic  
export GOOGLE_API_KEY="your-key"        # Google
export MISTRAL_API_KEY="your-key"       # Mistral
export OPENROUTER_API_KEY="your-key"    # OpenRouter
```

### Dependencies
All dependencies are managed with `uv`. Run examples with:

```bash
uv run python examples/example_name.py
```

## 🎯 Usage Patterns

### Modern API Usage
```python
from yamllm.core.llm import LLM

# Simple usage
llm = LLM(config_path="my_config.yaml")
response = llm.get_response("Hello!")

# Streaming usage  
for chunk in llm.get_streaming_response("Explain AI"):
    print(chunk, end='', flush=True)
```

### Error Handling
```python
from yamllm.core.exceptions import ProviderError, ToolExecutionError

try:
    response = llm.get_response("Calculate 2+2")
except ProviderError as e:
    print(f"Provider error: {e}")
except ToolExecutionError as e:
    print(f"Tool error: {e.message}")
```

## 🔧 Development

### Running Examples
```bash
# From project root
uv run python examples/cli_showcase.py
uv run python examples/openai/openai_example.py
```

### Creating Custom Examples
Use the modern patterns shown in `modern_yamllm_usage.py` as a template.

## 📚 Further Reading

- [Configuration Guide](../docs/configuration.md)
- [Tools Documentation](../docs/tools.md)
- [Provider Setup](../docs/providers.md)
- [CLI Reference](../CLAUDE.md#common-development-commands)

---

💡 **Tip**: Run `yamllm --help` to see all available CLI commands, or `yamllm <command> --help` for specific help.