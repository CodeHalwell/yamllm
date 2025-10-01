# YAMLLM CLI Reference

Complete reference for the YAMLLM command-line interface.

## Installation

```bash
# Install from source
pip install -e .

# Verify installation
yamllm --version
```

---

## Global Options

```bash
yamllm --help                 # Show help message
yamllm --version             # Show version information
```

---

## Commands Overview

### Setup & Configuration
- [`init`](#yamllm-init) - Interactive setup wizard
- [`config create`](#yamllm-config-create) - Create configuration file
- [`config validate`](#yamllm-config-validate) - Validate configuration
- [`config presets`](#yamllm-config-presets) - List config presets
- [`config models`](#yamllm-config-models) - List available models

### Chat & Interaction
- [`chat`](#yamllm-chat) - Interactive chat session
- [`run`](#yamllm-run) - Alias for chat
- [`quickstart`](#yamllm-quickstart) - Quick start guide
- [`guide`](#yamllm-guide) - Comprehensive getting started guide

### Tools
- [`tools list`](#yamllm-tools-list) - List available tools
- [`tools info`](#yamllm-tools-info) - Tool information
- [`tools test`](#yamllm-tools-test) - Test a tool
- [`tools manage`](#yamllm-tools-manage) - Interactive tool management
- [`tools search`](#yamllm-tools-search) - Search for tools

### System
- [`status`](#yamllm-status) - System status and health
- [`providers`](#yamllm-providers) - List LLM providers
- [`diagnose`](#yamllm-diagnose) - Environment diagnostics
- [`migrate-index`](#yamllm-migrate-index) - FAISS vector store migration

### UI & Themes
- [`theme list`](#yamllm-theme-list) - List themes
- [`theme preview`](#yamllm-theme-preview) - Preview theme
- [`theme set`](#yamllm-theme-set) - Set active theme
- [`theme current`](#yamllm-theme-current) - Show current theme
- [`theme reset`](#yamllm-theme-reset) - Reset to default

### MCP (Model Context Protocol)
- [`mcp list`](#yamllm-mcp-list) - List MCP connector tools

---

## Detailed Command Reference

### `yamllm init`

Interactive setup wizard for new users. Creates a configuration file and sets up environment.

**Usage:**
```bash
yamllm init
```

**What it does:**
1. Checks for existing configuration
2. Prompts for provider selection (OpenAI, Anthropic, Google, etc.)
3. Guides API key setup
4. Creates config file at specified location
5. Tests connection

**Example:**
```bash
$ yamllm init
Welcome to YAMLLM Setup Wizard!

Select your LLM provider:
1) OpenAI
2) Anthropic (Claude)
3) Google (Gemini)
4) Mistral
5) DeepSeek

Your choice: 1

Enter your OpenAI API key (or press Enter to use $OPENAI_API_KEY):
...
```

---

### `yamllm chat`

Start an interactive chat session with the LLM.

**Usage:**
```bash
yamllm chat [options]
```

**Options:**
- `--config PATH` - Path to configuration file (default: looks in common locations)
- `--provider NAME` - Override provider from config
- `--model NAME` - Override model from config
- `--system-prompt TEXT` - Override system prompt
- `--no-memory` - Disable conversation memory
- `--debug` - Enable debug logging

**Examples:**
```bash
# Basic chat
yamllm chat

# Chat with specific config
yamllm chat --config my-config.yaml

# Chat with debug logging
yamllm chat --debug

# Chat without memory (stateless)
yamllm chat --no-memory

# Override model
yamllm chat --model gpt-4o --provider openai
```

**Interactive Commands:**
```
/help       - Show help
/exit       - Exit chat
/quit       - Exit chat
/clear      - Clear screen
/history    - Show conversation history
/reset      - Reset conversation
/save       - Save conversation
/tools      - List available tools
/debug      - Toggle debug mode
```

---

### `yamllm run`

Alias for `yamllm chat`. Quick way to start a chat session.

**Usage:**
```bash
yamllm run [options]
```

See [`yamllm chat`](#yamllm-chat) for options.

---

### `yamllm config create`

Create a new configuration file with guided setup.

**Usage:**
```bash
yamllm config create [options]
```

**Options:**
- `--output PATH` - Output file path (default: `config.yaml`)
- `--preset NAME` - Use a preset template (minimal, full, tools, memory)
- `--provider NAME` - Provider to configure
- `--interactive` - Interactive mode with prompts

**Examples:**
```bash
# Create minimal config
yamllm config create --preset minimal --output my-config.yaml

# Interactive config creation
yamllm config create --interactive

# Create config for specific provider
yamllm config create --provider anthropic --preset full
```

**Presets:**
- `minimal` - Basic configuration without tools or memory
- `full` - Complete configuration with all options
- `tools` - Configuration with commonly used tools
- `memory` - Configuration with memory and vector store

---

### `yamllm config validate`

Validate a configuration file for syntax and completeness.

**Usage:**
```bash
yamllm config validate <config-file>
```

**Options:**
- `--strict` - Enable strict validation (check all optional fields)
- `--fix` - Attempt to fix common issues
- `--explain` - Show detailed explanation of errors

**Examples:**
```bash
# Basic validation
yamllm config validate config.yaml

# Strict validation with explanations
yamllm config validate config.yaml --strict --explain

# Validate and fix
yamllm config validate config.yaml --fix
```

**Output:**
```bash
✓ Configuration is valid
✓ Provider: openai
✓ Model: gpt-4o-mini
✓ Tools: 5 enabled
⚠ Warning: No memory configured
ℹ Suggestion: Consider adding retry configuration
```

---

### `yamllm config presets`

List available configuration presets and templates.

**Usage:**
```bash
yamllm config presets
```

**Output:**
```bash
Available Configuration Presets:

minimal              Basic configuration without tools or memory
full                 Complete configuration with all options  
tools                Configuration with commonly used tools
memory               Configuration with memory and vector store
research             Optimized for research with web search
coding               Optimized for coding assistance
creative             Optimized for creative writing
```

---

### `yamllm config models`

List available models for each provider.

**Usage:**
```bash
yamllm config models [provider]
```

**Options:**
- `provider` - Filter by provider (openai, anthropic, google, etc.)

**Examples:**
```bash
# List all models
yamllm config models

# List OpenAI models only
yamllm config models openai
```

**Output:**
```bash
OpenAI Models:
  gpt-4o              Latest GPT-4 Optimized
  gpt-4o-mini         Faster GPT-4 Optimized
  gpt-4-turbo         GPT-4 Turbo
  gpt-3.5-turbo       Fast and cost-effective

Anthropic Models:
  claude-3-5-sonnet-20241022    Latest Claude 3.5 Sonnet
  claude-3-opus-20240229        Most capable Claude 3
  claude-3-haiku-20240307       Fastest Claude 3
```

---

### `yamllm tools list`

List all available tools and tool packs.

**Usage:**
```bash
yamllm tools list [options]
```

**Options:**
- `--category NAME` - Filter by category (network, math, utility, file)
- `--search QUERY` - Search by keyword
- `--json` - Output in JSON format

**Examples:**
```bash
# List all tools
yamllm tools list

# List network tools
yamllm tools list --category network

# Search for tools
yamllm tools list --search "convert"

# JSON output
yamllm tools list --json
```

**Output:**
```bash
Available Tools (22 total):

Network & Web Tools:
  web_search          Search the internet using DuckDuckGo
  weather             Get weather information
  web_scraper         Scrape website content
  web_headlines       Get news headlines
  url_metadata        Get URL metadata

Math & Conversion:
  calculator          Safe arithmetic calculator
  unit_converter      Convert between units
  timezone            Convert between timezones

Utility Tools:
  datetime            Get current date/time
  uuid                Generate UUIDs
  hash                Generate hashes (MD5, SHA256)
  json                Validate and format JSON
  regex_extract       Extract patterns with regex
  ...
```

---

### `yamllm tools info`

Show detailed information about a specific tool.

**Usage:**
```bash
yamllm tools info <tool-name>
```

**Examples:**
```bash
yamllm tools info calculator
yamllm tools info web_search
```

**Output:**
```bash
Tool: calculator
Category: Math
Security: Safe (sandboxed evaluation)

Description:
  Executes arithmetic and mathematical operations safely.
  Supports +, -, *, /, **, %, sqrt, sin, cos, tan, log, etc.

Parameters:
  expression (string, required)
    The mathematical expression to evaluate
    Example: "2 + 2 * 3"

Usage Example:
  expression: "sqrt(16) + 10"
  result: 14.0

Security Notes:
  - No access to file system
  - No network access
  - Limited to mathematical operations
  - Safe eval implementation
```

---

### `yamllm tools test`

Test a tool with sample input.

**Usage:**
```bash
yamllm tools test <tool-name> [options]
```

**Options:**
- `--param KEY=VALUE` - Tool parameters
- `--interactive` - Interactive parameter entry

**Examples:**
```bash
# Test calculator
yamllm tools test calculator --param expression="2+2"

# Test weather (interactive)
yamllm tools test weather --interactive

# Test with multiple parameters
yamllm tools test unit_converter \
  --param value=100 \
  --param from_unit=celsius \
  --param to_unit=fahrenheit
```

---

### `yamllm tools manage`

Interactive tool management interface.

**Usage:**
```bash
yamllm tools manage
```

**Features:**
- Enable/disable tools
- Configure tool settings
- View tool usage statistics
- Test tools interactively

---

### `yamllm tools search`

Search for tools by keyword or description.

**Usage:**
```bash
yamllm tools search <query>
```

**Examples:**
```bash
yamllm tools search "convert"
yamllm tools search "web"
yamllm tools search "time"
```

---

### `yamllm status`

Show system status and health checks.

**Usage:**
```bash
yamllm status [options]
```

**Options:**
- `--verbose` - Show detailed information
- `--json` - Output in JSON format

**Output:**
```bash
YAMLLM System Status:

Version:              0.1.12
Python:               3.12.3
Installation:         /path/to/yamllm

Configuration:
  Config file:        ✓ Found
  Valid:              ✓ Yes
  Provider:           openai
  Tools enabled:      5

Environment:
  OPENAI_API_KEY:     ✓ Set
  ANTHROPIC_API_KEY:  ✗ Not set
  GOOGLE_API_KEY:     ✓ Set

Memory:
  Database:           ✓ Connected
  Conversations:      23
  Vector store:       ✓ Initialized
  Embeddings:         1,234

System:
  Disk space:         45.2 GB free
  Memory:             8.1 GB / 16 GB
  CPU:                4 cores
```

---

### `yamllm providers`

List all supported LLM providers and their status.

**Usage:**
```bash
yamllm providers [options]
```

**Options:**
- `--available` - Show only configured providers
- `--verbose` - Show detailed information

**Output:**
```bash
Supported LLM Providers:

✓ OpenAI
  Models: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo
  Features: Streaming, Tools, Vision
  API Key: Configured ✓

✗ Anthropic (Claude)
  Models: claude-3-5-sonnet, claude-3-opus, claude-3-haiku
  Features: Streaming, Tools, Vision
  API Key: Not configured

✓ Google (Gemini)
  Models: gemini-1.5-pro, gemini-1.5-flash
  Features: Streaming, Tools, Vision
  API Key: Configured ✓

...
```

---

### `yamllm diagnose`

Run diagnostics to check environment and configuration.

**Usage:**
```bash
yamllm diagnose [options]
```

**Options:**
- `--fix` - Attempt to fix common issues
- `--verbose` - Show detailed diagnostic information

**Output:**
```bash
Running YAMLLM Diagnostics...

✓ Python version: 3.12.3 (compatible)
✓ Package installation: OK
✓ Dependencies: All installed

Configuration:
  ✓ Config file found
  ✓ YAML syntax valid
  ✓ Schema validation passed
  
Environment:
  ✓ API keys found (2/3 providers)
  ⚠ ANTHROPIC_API_KEY not set
  ✓ .env file found
  
Memory:
  ✓ Database accessible
  ✓ Vector store initialized
  ⚠ Vector store dimension mismatch (expected 1536, got 768)
  
Network:
  ✓ Internet connectivity
  ✓ API endpoints reachable
  
Permissions:
  ✓ Read/write access to memory directory
  ✓ Read/write access to logs directory

Issues Found: 2
  ⚠ Vector store dimension mismatch
    Fix: Run 'yamllm migrate-index' to rebuild vector store
  
  ⚠ ANTHROPIC_API_KEY not set
    Fix: Add ANTHROPIC_API_KEY to your .env file
```

---

### `yamllm migrate-index`

Inspect or rebuild FAISS vector store index.

**Usage:**
```bash
yamllm migrate-index [options]
```

**Options:**
- `--config PATH` - Configuration file
- `--rebuild` - Rebuild the index
- `--purge` - Delete and recreate empty index
- `--backup` - Create backup before migration

**Examples:**
```bash
# Inspect index
yamllm migrate-index --config config.yaml

# Rebuild index
yamllm migrate-index --config config.yaml --rebuild --backup

# Purge and recreate
yamllm migrate-index --config config.yaml --purge
```

**Warning:** Use `--purge` with caution as it deletes all embeddings.

---

### `yamllm theme list`

List available UI themes.

**Usage:**
```bash
yamllm theme list
```

**Output:**
```bash
Available Themes:

  default             Default YAMLLM theme
  nord                Nord color scheme
  dracula             Dracula dark theme
  solarized-dark      Solarized Dark
  solarized-light     Solarized Light
  monokai             Monokai theme
  
Current theme: default
```

---

### `yamllm theme preview`

Preview a theme in the terminal.

**Usage:**
```bash
yamllm theme preview <theme-name>
```

**Examples:**
```bash
yamllm theme preview nord
yamllm theme preview dracula
```

---

### `yamllm theme set`

Set the active theme.

**Usage:**
```bash
yamllm theme set <theme-name>
```

**Examples:**
```bash
yamllm theme set nord
yamllm theme set dracula
```

---

### `yamllm theme current`

Show the currently active theme.

**Usage:**
```bash
yamllm theme current
```

---

### `yamllm theme reset`

Reset to default theme.

**Usage:**
```bash
yamllm theme reset
```

---

### `yamllm mcp list`

List tools available from configured MCP connectors.

**Usage:**
```bash
yamllm mcp list [options]
```

**Options:**
- `--config PATH` - Configuration file
- `--verbose` - Show detailed information

---

### `yamllm quickstart`

Display quick start guide.

**Usage:**
```bash
yamllm quickstart
```

Shows a condensed guide to get started quickly.

---

### `yamllm guide`

Display comprehensive getting started guide.

**Usage:**
```bash
yamllm guide
```

Shows detailed step-by-step instructions for setting up and using YAMLLM.

---

## Environment Variables

YAMLLM respects the following environment variables:

### API Keys
```bash
OPENAI_API_KEY          # OpenAI API key
ANTHROPIC_API_KEY       # Anthropic (Claude) API key
GOOGLE_API_KEY          # Google (Gemini) API key
MISTRAL_API_KEY         # Mistral API key
DEEPSEEK_API_KEY        # DeepSeek API key
AZURE_OPENAI_KEY        # Azure OpenAI key
OPENWEATHER_API_KEY     # Weather API key (optional)
```

### Configuration
```bash
YAMLLM_CONFIG           # Default config file path
YAMLLM_LOG_LEVEL        # Log level (DEBUG, INFO, WARNING, ERROR)
YAMLLM_THEME            # Default UI theme
YAMLLM_NO_COLOR         # Disable colored output (any value)
```

### Behavior
```bash
YAMLLM_DISABLE_TELEMETRY    # Disable telemetry (any value)
YAMLLM_CACHE_DIR            # Cache directory location
YAMLLM_DATA_DIR             # Data directory location
```

---

## Configuration Files

YAMLLM looks for configuration files in the following locations (in order):

1. Path specified with `--config` flag
2. `$YAMLLM_CONFIG` environment variable
3. `./config.yaml` (current directory)
4. `./.yamllm.yaml` (current directory)
5. `~/.yamllm/config.yaml` (home directory)
6. `~/.config/yamllm/config.yaml` (XDG config)

---

## Debugging

### Enable Debug Logging

```bash
# Via environment variable
export YAMLLM_LOG_LEVEL=DEBUG
yamllm chat

# Via command line
yamllm chat --debug

# Via config file
logging:
  level: "DEBUG"
  console: true
```

### View Logs

```bash
# Default log location
tail -f logs/yamllm.log

# Or if configured differently
tail -f path/to/your/logfile.log
```

### Diagnose Issues

```bash
# Run full diagnostics
yamllm diagnose --verbose

# Test specific tool
yamllm tools test <tool-name> --interactive

# Validate config
yamllm config validate config.yaml --explain
```

---

## Tips & Tricks

### Quick Chat with Different Models

```bash
# Try GPT-4o
yamllm chat --model gpt-4o

# Try Claude
yamllm chat --provider anthropic --model claude-3-5-sonnet-20241022

# Try Gemini
yamllm chat --provider google --model gemini-1.5-flash
```

### Create Project-Specific Configs

```bash
# Create project directory
mkdir my-project && cd my-project

# Create minimal config
yamllm config create --preset minimal

# Start chatting
yamllm chat
```

### Test Tools Before Use

```bash
# List all tools
yamllm tools list

# Get tool info
yamllm tools info calculator

# Test the tool
yamllm tools test calculator --param expression="2+2"
```

### Export Conversation History

```python
# In Python
from yamllm.memory import ConversationStore

store = ConversationStore("memory/conversation_history.db")
messages = store.get_messages()

# Export to JSON
import json
with open("history.json", "w") as f:
    json.dump(messages, f, indent=2)
```

---

## Common Issues

### "No configuration file found"

**Solution:**
```bash
# Create a config file
yamllm config create --preset minimal --output config.yaml

# Or specify path explicitly
yamllm chat --config /path/to/config.yaml
```

### "API key not found"

**Solution:**
```bash
# Set environment variable
export OPENAI_API_KEY="your-key-here"

# Or add to .env file
echo "OPENAI_API_KEY=your-key-here" >> .env
```

### "Tool execution timeout"

**Solution:**
```yaml
# Increase timeout in config.yaml
tools:
  tool_timeout: 60  # Increase from default 10 seconds
```

### "Vector store dimension mismatch"

**Solution:**
```bash
# Rebuild vector store
yamllm migrate-index --rebuild --backup
```

---

## Getting Help

### In-App Help

```bash
yamllm --help                    # Global help
yamllm <command> --help          # Command-specific help
yamllm tools info <tool>         # Tool documentation
yamllm guide                     # Full guide
```

### Documentation

- [README.md](../README.md) - Overview and quick start
- [COMPREHENSIVE_REVIEW.md](../COMPREHENSIVE_REVIEW.md) - Detailed analysis
- [docs/configuration.md](configuration.md) - Configuration reference
- [docs/tools.md](tools.md) - Tools documentation

### Community

- GitHub Issues: Bug reports and feature requests
- GitHub Discussions: Questions and community help

---

**Last Updated:** December 2024  
**Version:** 0.1.12
