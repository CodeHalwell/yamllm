# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Linting and Formatting
- `ruff check .` - Lint code with ruff
- `black --check .` - Check code formatting with black
- `black .` - Format code with black
- `mypy yamllm` - Type check the yamllm package

### Testing
- `pytest` - Run all tests
- `pytest -q` - Run tests quietly
- `pytest tests/test_specific.py` - Run specific test file
- `pytest -k "test_name"` - Run tests matching pattern

### Installation and Dependencies
- `pip install -e .` - Install package in development mode
- `uv add package-name` - Add dependency using uv

## Architecture Overview

YAMLLM is a Python library for YAML-based LLM configuration and execution with a provider-agnostic interface.

### Core Structure

- **Core Module** (`yamllm/core/`): Contains the main LLM interface and provider abstractions
  - `llm.py`: Main LLM class with unified interface
  - `parser.py`: YAML configuration parsing
  - `providers/`: Provider implementations for different LLM services

- **Provider System**: Supports multiple LLM providers through a unified interface:
  - OpenAI (GPT models)
  - Anthropic (Claude models) 
  - Google (Gemini models)
  - Mistral
  - DeepSeek (OpenAI-compatible)
  - Azure OpenAI
  - Azure AI Foundry
  - OpenRouter

- **Tools System** (`yamllm/tools/`): Extensible tool system for function calling
  - `manager.py`: Tool management and execution
  - `utility_tools.py`: Built-in tools (web search, calculator, weather, etc.)
  - Supports tool packs (common, web, files, crypto, etc.)

- **Memory System** (`yamllm/memory/`): Conversation history and vector storage
  - SQLite-based conversation storage
  - FAISS vector store for semantic search
  - Configurable memory management

### Provider Architecture

All providers implement the `BaseProvider` interface with standardized methods:
- `get_completion()`: Single completion
- `get_streaming_completion()`: Streaming completion
- `create_embedding()`: Embedding creation
- `format_tool_calls()` / `format_tool_results()`: Tool use formatting

Providers are created via `ProviderFactory.create_provider()` based on configuration.

### Configuration System

YAMLLM uses YAML configuration files with these required sections:
- `provider`: LLM service configuration
- `model_settings`: Model parameters (temperature, max_tokens, etc.)
- `request`: Request settings (timeout, retry logic)
- `context`: System prompt and memory settings
- `output`: Response formatting
- `logging`: Logging configuration
- `tools`: Tool enablement and configuration
- `safety`: Content filtering and rate limits

Configuration supports environment variable substitution using `${VAR_NAME}` syntax.

### Tool System Design

Tools follow a standardized interface with:
- Tool registration in `tool_manager.py`
- Automatic conversion between provider-specific tool formats
- Support for both individual tools and tool packs
- Model Context Protocol (MCP) connector support

### Testing Strategy

Tests are organized by component:
- Provider-specific tests (`test_*_provider.py`)
- Core functionality tests (`test_llm.py`, `test_config.py`)
- Tool system tests (`test_tool_*.py`)
- Integration tests for memory and streaming

Use `pytest` for running tests. The CI pipeline runs tests on Python 3.10 and 3.11.

### Development Notes

- The codebase uses type hints extensively - ensure new code includes proper typing
- All providers must implement the `BaseProvider` interface consistently
- Configuration validation happens in `parser.py` using Pydantic models
- Memory management is automatic but configurable per session
- Tool execution happens in separate threads with timeout protection