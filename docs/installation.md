[← Back to Index](index.md)

# Installation Guide

## Prerequisites

- Python 3.10 or higher
- pip (Python package manager) or uv (recommended)
- Git (for development installation)

## Quick Installation

Using pip:
```bash
pip install yamllm-core
```

Using uv (recommended):
```bash
uv pip install yamllm-core
```

## Development Installation

1. Clone the repository:
```bash
git clone https://github.com/codehalwell/yamllm.git
cd yamllm
```

2. Create and activate a virtual environment:
```bash
# Using uv (recommended)
uv venv
.venv/Scripts/activate

# Using venv
python -m venv .venv
.venv/Scripts/activate
```

3. Install development dependencies:
```bash
# Using uv
uv pip install -e ".[dev]"

# Using pip
pip install -e ".[dev]"
```

## Configuration

1. Create a `.env` file in your project root:
```plaintext
# Required API keys (use at least one)
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here
MISTRAL_API_KEY=your_mistral_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here
```

2. Create a configuration file (config.yaml):
```yaml
provider:
  name: "mistralai"  # or openai, google, deepseek
  model: "mistral-small-latest"
  api_key: ${MISTRAL_API_KEY}

model_settings:
  temperature: 0.7
  max_tokens: 1000
```

## Verifying Installation

```python
from yamllm.core.llm import MistralAI
import os
import dotenv

dotenv.load_dotenv()

# Initialize LLM
llm = MistralAI(
    config_path="config.yaml",
    api_key=os.environ.get("MISTRAL_API_KEY")
)

# Test query
response = llm.query("Hello, how are you?")
```

## System Requirements

- Operating System:
  - Windows 10/11
  - macOS 10.15 or higher
  - Linux (major distributions)
- Memory: 4GB RAM minimum
- Storage: 100MB free space
- CPU: x86_64 architecture

## Troubleshooting

### Common Issues

1. Permission Errors:
```bash
# Windows (Run as Administrator)
pip install --user yamllm-core

# Linux/macOS
pip install --user yamllm-core
```

2. SSL Certificate Errors:
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org yamllm-core
```

3. Virtual Environment Issues:
```bash
# Windows
python -m pip install --upgrade pip
python -m pip install virtualenv
python -m virtualenv .venv

# Linux/macOS
python3 -m pip install --upgrade pip
python3 -m pip install virtualenv
python3 -m virtualenv .venv
```

## Next Steps

- Read the [Configuration Guide](configuration.md)
- Try the [Examples](examples.md)
- Check the [API Reference](api/index.md)

## Support

If you encounter any issues:
1. Check our [GitHub Issues](https://github.com/codehalwell/yamllm/issues)
2. Join our [Discord Community](https://discord.gg/yamllm)
3. Read the [FAQ](faq.md)