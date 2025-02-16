# YAMLLM

A Python library for YAML-based LLM configuration and execution.

## Installation

```bash
pip install yamllm-core
```

## Quick Start

```python
from yamllm import LLM
import os

# Initialize LLM with config
llm = LLM(config_path="path/to/config.yaml")
llm.api_key = os.environ.get("OPENAI_API_KEY")

# Make a query
response = llm.query("What is the meaning of life?")
print(response)
```

## Configuration
YAMLLM uses YAML files for configuration. Set up a `.config` file to define the parameters for your LLM instance. This file should include settings such as the model type, temperature, maximum tokens, and system prompt.

Example configuration:

```yaml
model: gpt-4-turbo-preview
temperature: 0.7
max_tokens: 500
system_prompt: "You are a helpful AI assistant."
```

Place the `.config` file in your project directory and reference it in your code to initialize the LLM instance.

Example configuration:

```yaml
model: gpt-4-turbo-preview
temperature: 0.7
max_tokens: 500
system_prompt: "You are a helpful AI assistant."
```

## Features

- YAML-based configuration
- Simple API interface
- Customizable prompt templates
- Error handling and retry logic

## License

MIT License