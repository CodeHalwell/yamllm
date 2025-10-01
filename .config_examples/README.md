# YAMLLM Configuration Examples

This directory contains example configuration files to help you get started with YAMLLM.

## Quick Start Configs

### Minimal Configurations (No Tools, No Memory)

These are the simplest configurations for basic usage:

- **`openai_minimal.yaml`** - OpenAI GPT-4o-mini with basic settings
- **`anthropic_minimal.yaml`** - Anthropic Claude 3.5 Sonnet with basic settings
- **`google_minimal.yaml`** - Google Gemini 1.5 Flash with basic settings

**Usage:**
```python
from yamllm import OpenAIGPT
import os

llm = OpenAIGPT(
    config_path=".config_examples/openai_minimal.yaml",
    api_key=os.environ.get("OPENAI_API_KEY")
)
response = llm.query("Hello!")
```

### Advanced Configuration

- **`with_tools.yaml`** - Full configuration showing tool usage, memory, and all options

## Environment Variables

All configurations expect API keys to be set as environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Google
export GOOGLE_API_KEY="..."

# Mistral
export MISTRAL_API_KEY="..."

# DeepSeek
export DEEPSEEK_API_KEY="sk-..."

# Weather API (optional, for weather tool)
export OPENWEATHER_API_KEY="..."
```

You can also use a `.env` file in your project root:

```bash
# .env file
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

## Configuration Sections

### Provider
Specifies which LLM provider to use:
```yaml
provider:
  name: "openai"  # openai, anthropic, google, mistral, deepseek, azure
  model: "gpt-4o-mini"
  base_url: # Optional: custom endpoint
```

### Model Settings
Controls generation parameters:
```yaml
model_settings:
  temperature: 0.7  # 0.0 to 2.0
  max_tokens: 1000
  top_p: 1.0
  frequency_penalty: 0.0
  presence_penalty: 0.0
```

### Context
Manages conversation context and memory:
```yaml
context:
  system_prompt: "You are a helpful assistant."
  max_context_length: 16000
  memory:
    enabled: true
    max_messages: 10
    session_id: "my-session"
```

### Tools
Enables and configures tool usage:
```yaml
tools:
  enabled: true
  tool_timeout: 30
  tool_list:
    - calculator
    - web_search
    - weather
```

See [docs/tools.md](../docs/tools.md) for the complete list of 22 available tools.

### Output
Controls response formatting:
```yaml
output:
  format: "text"  # text, json, markdown
  stream: false
```

### Logging
Configures logging behavior:
```yaml
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  console: true
  file: "logs/yamllm.log"
```

## Customization Tips

1. **Start Simple:** Begin with a minimal config and add features as needed
2. **Test Tools:** Enable one tool at a time to understand behavior
3. **Debug Mode:** Set `logging.level: "DEBUG"` to see detailed information
4. **Memory:** Start with `memory.enabled: false` for simple queries
5. **Streaming:** Enable `output.stream: true` for real-time responses

## Common Patterns

### Chat Application
```yaml
context:
  memory:
    enabled: true
    max_messages: 20
output:
  stream: true
```

### Data Processing
```yaml
tools:
  enabled: true
  tool_list:
    - file_read
    - csv_preview
    - json
context:
  memory:
    enabled: false
```

### Research Assistant
```yaml
tools:
  enabled: true
  tool_list:
    - web_search
    - web_scraper
    - web_headlines
    - calculator
context:
  memory:
    enabled: true
    vector_store:
      enabled: true
```

## Troubleshooting

### "Configuration validation error"
- Check YAML syntax (indentation matters!)
- Ensure all required fields are present
- Verify model names are correct for your provider

### "API key not found"
- Set environment variables before running
- Check `.env` file location
- Use absolute paths if needed

### "Tool execution timeout"
- Increase `tools.tool_timeout`
- Check network connectivity
- Enable DEBUG logging to see details

## Next Steps

1. Copy a minimal config to your project
2. Set your API key as an environment variable
3. Run a simple query to test
4. Enable tools and memory as needed
5. Read [docs/configuration.md](../docs/configuration.md) for advanced options

## Security Note

⚠️ **Never commit API keys to version control!**

Always use environment variables or `.env` files (and add `.env` to `.gitignore`).
