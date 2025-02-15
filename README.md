# YAMLLM

A Python package for easily configuring and interacting with Large Language Models (LLMs) using YAML configuration files.

## Features

- Simple YAML-based configuration for LLM settings
- Built-in support for multiple LLM providers (OpenAI, etc.)
- Extensible tool system for custom functionality
- Conversation memory management
- Advanced ML tools integration
- Built-in file handling and YAML processing

## Installation

Install using pip:

```sh
pip install yamllm
```

## Quick Start

1. Create a configuration file [`config.yaml`](yamllm/src/yamllm/llm.py):

```yaml
provider:
  name: "openai"
  model: "gpt-3.5-turbo"
  api_key: "your-api-key"

model_settings:
  temperature: 0.7
  max_tokens: 1000
```

2. Use YAMLLM in your code:

```python
from yamllm import LLM

llm = LLM(config_path="config.yaml")
response = llm.query("Hello, how are you?")
print(response)
```

## Configuration

YAMLLM uses a comprehensive YAML configuration system. Key sections include:

- Provider settings
- Model parameters
- Request handling
- Context management
- Memory settings
- Output formatting
- Tool configuration
- Logging
- Safety settings

See [`yamllm/examples/basic_config.yaml`](yamllm/examples/basic_config.yaml) for a complete example.

## Tools

YAMLLM includes several built-in tools:

- File Tools
  - `ReadFileContent`: Read file contents
  - `WriteFileContent`: Write to files

- YAML Tools
  - `ParseYAML`: Parse YAML strings
  - `DumpYAML`: Convert objects to YAML

- ML Tools
  - `DataLoader`: Load data from various formats
  - `EDAAnalyzer`: Perform exploratory data analysis
  - `DataPreprocessor`: Clean and preprocess data
  - `ModelTrainer`: Train and evaluate ML models

- Utility Tools
  - `Calculator`: Perform calculations
  - `WebSearch`: Search the web
  - `TimezoneTool`: Convert between timezones
  - `UnitConverter`: Convert between units

## Development

Requirements:
- Python 3.12+
- Dependencies listed in [`yamllm/pyproject.toml`](yamllm/pyproject.toml)

Setup development environment:

```sh
pip install -e .
```

Run tests:

```sh
python -m unittest discover src/tests
```

## License

MIT License - See [`yamllm/LICENSE`](yamllm/LICENSE) for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Submit a pull request

## Support

- Report issues on GitHub
- Check documentation in the code
- See example configurations in examples/