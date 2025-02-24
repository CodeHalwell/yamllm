# YAMLLM Documentation

YAMLLM is a Python library for YAML-based LLM configuration and execution.

## Quick Links

- [Installation Guide](installation.md)
- [Configuration Guide](configuration.md)
- [API Reference](api/index.md)
- [Examples](examples.md)
- [Contributing](contributing.md)

## Overview

YAMLLM provides a flexible framework for configuring and executing Language Model (LLM) interactions using YAML configuration files. It supports multiple LLM providers including OpenAI, Google, DeepSeek, and MistralAI. The YAML configuration files allow you to store base settings and experiment without needing multiple changes to the .py files.

The aim of the package is to allow a unique and quick to set up environment, therefore, the rich library has been used to allow for CLI interactions that are easy to use.

Any code snipits etc are printed in markdown format so can be easily read and copied to your editor.

## Features

- YAML-based configuration
- Simple API interface
- Customizable prompt templates in a YAML file
- Easy to use/read CLI interface available using rich
- Error handling and retry logic
- Built-in memory management using SQLite
- Vector database for long-term memory using semantic search
- Support for both streaming and non-streaming responses

## License

MIT License - see [LICENSE](https://github.com/codehalwell/yamllm/blob/main/LICENSE) for details.