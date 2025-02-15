# Usage Guide

## Basic Usage

```python
from yamllm import LLM
import os

# Initialize LLM with config
llm = LLM(config_path="config.yaml")
llm.api_key = os.environ.get("OPENAI_API_KEY")

# Make a query
response = llm.query("What is YAMLLM?")
print(response)
```

## Configuration

Create a `config.yaml` file:

```yaml
model: gpt-4-turbo-preview
temperature: 0.7
max_tokens: 500
system_prompt: "You are a helpful AI assistant."
```