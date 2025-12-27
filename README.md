# YAMLLM

**The Universal LLM Terminal** - A Python library for YAML-based LLM configuration and execution.

> Chat with any LLM (OpenAI, Claude, Gemini, Mistral, and more) using one beautiful interface. Switch models mid-conversation. Keep your workflow, change your provider.

## Installation

### For Development

```bash
# Clone the repository
git clone https://github.com/CodeHalwell/yamllm.git
cd yamllm

# Install in editable mode with pip
pip install -e .

# Or with uv (recommended)
uv pip install -e .
```

### From PyPI (Coming Soon)

> **Note:** PyPI package publication is planned for v1.0. For now, please install from source as shown above.

## Quick Start

### Using the CLI (Recommended)

The fastest way to get started is using the interactive CLI:

```bash
# Interactive setup wizard
yamllm init

# Start chatting (after setup)
yamllm chat --config config.yaml

# List available commands
yamllm --help

# Check system status
yamllm status

# List all available tools
yamllm tools list
```

### 1. Simple Query (Programmatic)

In order to run a simple query, create a script as follows. The library uses Rich to print responses in the console with nice formatting.

```python
from yamllm import OpenAIGPT, GoogleGemini, DeepSeek, MistralAI
import os
import dotenv

dotenv.load_dotenv()

config_path = "path/to/config.yaml"

# Initialize LLM with config
llm = GoogleGemini(config_path=config_path, api_key=os.environ.get("GOOGLE_API_KEY"))

# Make a query - response is automatically printed with Rich formatting
response = llm.query("Give me some boilerplate PyTorch code please")
```

### 2. Interactive Conversation

For an ongoing conversation with the model, use this pattern:

```python
from yamllm import OpenAIGPT, GoogleGemini, DeepSeek, MistralAI
from rich.console import Console
import os
import dotenv

dotenv.load_dotenv()
console = Console()

config_path = "path/to/config.yaml"

llm = GoogleGemini(config_path=config_path, api_key=os.environ.get("GOOGLE_API_KEY"))

while True:
    try:          
        prompt = input("\nHuman: ")
        if prompt.lower() == "exit":
            break
        
        response = llm.query(prompt)
        if response is None:
            continue
        
    except FileNotFoundError as e:
        console.print(f"[red]Configuration file not found:[/red] {e}")
    except ValueError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
    except Exception as e:
        console.print(f"[red]An error occurred:[/red] {str(e)}")
```

## CLI Commands Reference

yamllm provides a rich command-line interface for all operations:

### Chat Commands
```bash
yamllm chat --config config.yaml          # Start interactive chat
yamllm chat --config config.yaml --style compact  # Compact output style
```

### Configuration Commands
```bash
yamllm config create --provider openai    # Create new config
yamllm config presets                     # List available presets
yamllm config validate config.yaml        # Validate config file
yamllm config edit config.yaml            # Edit config (opens editor)
```

### Tool Commands
```bash
yamllm tools list                         # List all available tools
yamllm tools help <tool_name>             # Get help for specific tool
yamllm tools test <tool_name>             # Test a tool
```

### Memory Commands
```bash
yamllm memory list                        # List conversations
yamllm memory export <session_id>         # Export conversation
yamllm memory clear <session_id>          # Clear conversation
yamllm memory migrate                     # Migrate vector store
```

### System Commands
```bash
yamllm init                               # Interactive setup wizard
yamllm status                             # Check system health
yamllm providers                          # List supported providers
yamllm quick-start                        # Show quick start guide
```

## Working with Conversation History

You can view your conversation history using the ConversationStore class. This will display all messages in a tabulated format:

```python
from yamllm import ConversationStore
# Optional: install extras `viz` to use pandas/tabulate output
try:
    import pandas as pd
    from tabulate import tabulate
except Exception:  # pragma: no cover - docs example
    pd = None
    tabulate = None

# Initialize the conversation store
history = ConversationStore("yamllm/memory/conversation_history.db")

# Retrieve messages
messages = history.get_messages()

# Pretty print if optional deps available
if pd and tabulate:
    df = pd.DataFrame(messages)
    print(tabulate(df, headers='keys', tablefmt='psql'))
else:
    # Fallback: simple print
    for m in messages:
        print(f"{m['role']}: {m['content']}")
```

## Working with Vector Store

The vector store allows you to manage and inspect embedded vectors from your conversations:

```python
from yamllm.memory import VectorStore

# Initialize the vector store
vector_store = VectorStore()

# Retrieve vectors and metadata
vectors, metadata = vector_store.get_vec_and_text()

# Display vector store information
print(f"Number of vectors: {len(vectors)}")
print(f"Vector dimension: {vectors.shape[1] if len(vectors) > 0 else 0}")
print(f"Number of metadata entries: {len(metadata)}")
print(metadata)
```

## Tools

### Available Tools

YAMLLM integrates a comprehensive set of specialized tools to enhance functionality:

#### Network & Web Tools
- **WebSearch:** Fetches up-to-date information from the internet using DuckDuckGo API
- **Weather:** Retrieves current weather conditions and forecasts using OpenWeatherMap API
- **WebScraper:** Extracts data from websites using Beautiful Soup
- **WebHeadlines:** Fetches news headlines and summaries
- **URLMetadata:** Retrieves metadata from URLs

#### Math & Conversion Tools
- **Calculator:** Executes arithmetic and mathematical operations safely
- **UnitConverter:** Converts between different units (length, weight, temperature, etc.)
- **TimezoneTool:** Converts times between different timezones

#### Utility Tools
- **DateTimeTool:** Gets current date/time with formatting and offset options
- **UUIDTool:** Generates UUIDs
- **RandomStringTool:** Generates random strings
- **RandomNumberTool:** Generates random numbers
- **Base64EncodeTool:** Encodes text to Base64
- **Base64DecodeTool:** Decodes Base64 to text
- **HashTool:** Generates hashes (MD5, SHA256, etc.)
- **JSONTool:** Validates and formats JSON
- **RegexExtractTool:** Extracts patterns using regular expressions
- **LoremIpsumTool:** Generates Lorem Ipsum placeholder text

#### File Tools
- **FileReadTool:** Reads file contents securely
- **FileSearchTool:** Searches for files in directories
- **CSVPreviewTool:** Previews CSV file contents

#### Helper Tools
- **ToolsHelpTool:** Lists available tools and their descriptions

All tools are designed with security in mind, featuring path restrictions, domain blocking for network tools, and input sanitization.

## Configuration
YAMLLM uses YAML files for configuration. Set up a `.config` file to define the parameters for your LLM instance. This file should include settings such as the model type, temperature, maximum tokens, and system prompt.

Example configuration:

```yaml
  # LLM Provider Settings
provider:
  name: "openai"  # supported: openai, google, deepseek and mistralai supported.
  model: "gpt-4o-mini"  # model identifier
  # Do NOT store API keys in config files. Provide the key via environment
  # variables and pass it into the LLM constructor.
  base_url: # optional: for custom endpoints when using the google, deepseek or mistral

# Model Configuration
model_settings:
  temperature: 0.7
  max_tokens: 1000
  top_p: 1.0
  frequency_penalty: 0.0
  presence_penalty: 0.0
  stop_sequences: []
  
# Request Settings
request:
  timeout: 30  # seconds
  retry:
    max_attempts: 3
    initial_delay: 1
    backoff_factor: 2
    
# Context Management
context:
  system_prompt: "You are a helpful, conversational assistant with access to tools. 
    When asked questions about current events, news, calculations, or unit conversions, use the appropriate tool.
    For current information, use the web_search tool instead of stating you don't have up-to-date information.

    Always present information in a natural, conversational way:
    - For web search results, summarize the key points in your own words
    - For calculations, explain the result in plain language
    - For conversions, provide context about the conversion
    - Use a friendly, helpful tone throughout

    Do not show raw data or JSON in your responses unless specifically asked to do so."
  max_context_length: 16000
  memory:
    enabled: true
    max_messages: 10  # number of messages to keep in conversation history
    session_id: "session2"
    conversation_db: "memory/conversation_history.db"
    vector_store:
      index_path: "memory/vector_store/faiss_index.idx"
      metadata_path: "memory/vector_store/metadata.pkl"
      top_k: 3
    
# Output Formatting
output:
  format: "text"  # supported: text, json, markdown
  stream: false

logging:
  level: "INFO"
  file: "yamllm.log"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Tool Management 
tools:
  enabled: true
  tool_timeout: 10  # seconds
  tool_list: 
    - calculator
    - web_search
    - weather
    - web_scraper
    - web_headlines
    - url_metadata
    - file_read
    - datetime
    - timezone
    - unit_converter
    # Add more tools as needed - see docs/tools.md for full list

# Safety Settings
safety:
  content_filtering: true
  max_requests_per_minute: 60
  sensitive_keywords: []
```

Place the `.config` file in your project directory and reference it in your code to initialize the LLM instance.

### Debugging and Logging

To enable detailed logging for debugging:

```yaml
logging:
  level: "DEBUG"  # Use DEBUG for development, INFO for production
  file: "yamllm.log"
  console: true  # Enable console output for logs
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

You can also set logging programmatically:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now your LLM calls will show detailed logs
llm = GoogleGemini(config_path=config_path, api_key=api_key)
```

For tool debugging, enable tool output visibility:

```python
# Set event callback to see tool execution
def on_event(event):
    print(f"Event: {event}")

llm.set_event_callback(on_event)
```

## Troubleshooting

### Common Issues

**"Module not found" errors:**
- Ensure you installed the package: `pip install -e .`
- Check you're in the correct virtual environment

**API Key errors:**
- Verify your API keys are set in environment variables
- Check `.env` file is in the correct location
- Never commit API keys to version control

**Configuration errors:**
- Validate your YAML syntax
- Check that all required fields are present
- Use DEBUG logging to see configuration parsing details

**Tool execution failures:**
- Check tool timeout settings in config
- Verify network connectivity for web tools
- Check file permissions for file tools
- See logs for specific error messages

### Getting Help

- Check [COMPREHENSIVE_REVIEW.md](COMPREHENSIVE_REVIEW.md) for known issues
- Review [docs/](docs/) for detailed documentation
- See [examples/](examples/) for working code samples
- Open an issue on GitHub for bugs or feature requests

## Features

### 🎯 Core Capabilities
- **Universal Provider Support**: Switch between OpenAI, Anthropic (Claude), Google (Gemini), Mistral, DeepSeek, Azure OpenAI, Azure AI Foundry, and OpenRouter
- **22 Built-in Tools**: Web search, weather, file operations, calculations, conversions, and more
- **Rich Terminal UI**: Beautiful themes with streaming support
- **Smart Memory**: SQLite conversation history + FAISS vector store for semantic search
- **MCP Integration**: Connect to external Model Context Protocol servers
- **Type-Safe Configuration**: YAML-based config with Pydantic validation

### 🛠️ Developer Experience
- Simple API interface with comprehensive error handling
- Streaming and non-streaming response modes
- Thread-safe tool execution with security controls
- Customizable prompt templates and system prompts
- Exponential backoff retry logic
- Event callbacks for tool execution tracking
- Environment variable support with dotenv

### 🎨 UI & Themes
- Rich console output with syntax highlighting
- Multiple theme support (default, synthwave, and more coming)
- Runtime theme switching
- Message bubbles with customizable styles
- Tool execution visualization
- Token usage tracking

## Why yamllm?

### 🌟 **The Only Multi-Provider LLM Terminal**

Unlike Claude Code (Anthropic only) or Gemini CLI (Google only), yamllm lets you:
- **Switch providers instantly** without changing your code or workflow
- **Compare responses** from different models side-by-side
- **Avoid vendor lock-in** by staying provider-agnostic
- **Use the best model for each task** - GPT-4 for reasoning, Claude for coding, Gemini for speed

### 📊 **Comparison with Alternatives**

| Feature | yamllm | Claude Code | Gemini CLI |
|---------|--------|-------------|------------|
| Multi-Provider | ✅ 8 providers | ❌ Anthropic only | ❌ Google only |
| Open Source | ✅ MIT | ✅ | ✅ Apache 2.0 |
| Built-in Tools | ✅ 22 tools | ✅ 30+ tools | ✅ Built-in |
| Memory System | ✅ SQLite + FAISS | ✅ | ⚠️ Basic |
| MCP Support | ✅ Client mode | ✅ | ⚠️ Limited |
| Custom Themes | ✅ Yes | ⚠️ Limited | ❌ |
| YAML Config | ✅ Yes | ✅ | ✅ |

### Project Status

This project is in active development. See [REVIEW_SUMMARY.md](REVIEW_SUMMARY.md) for the current status and [yamllm_manifesto.md](yamllm_manifesto.md) for the project vision. We're working toward a v1.0 release with ~60% of manifesto features currently implemented.

**Current Strengths:**
- ✅ **Best-in-class multi-provider architecture** (unique advantage)
- ✅ Robust configuration system with type safety
- ✅ Comprehensive 22-tool ecosystem with security controls
- ✅ Memory management with SQLite + FAISS vector store
- ✅ Good test coverage and code quality

**In Progress (Roadmap to v1.0):**
- ⚠️ Agentic loop framework for autonomous task completion
- ⚠️ Git integration for repository workflows
- ⚠️ Performance optimization (<350ms first token target)
- ⚠️ Expanded theme library (10+ themes planned)
- ⚠️ Additional developer tools (shell, code search, SQL)

## License

MIT License
