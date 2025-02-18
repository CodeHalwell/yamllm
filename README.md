# YAMLLM

A Python library for YAML-based LLM configuration and execution.

## Installation

```bash
pip install yamllm-core
```

## Quick Start

```python
from yamllm.core.llm import OpenAIGPT, GoogleGemini, DeepSeek
import os
import dotenv

dotenv.load_dotenv()

# Initialize LLM with config
llm = OpenAIGPT(config_path="path/to/config.yaml", api_key=os.environ.get("OPENAI_API_KEY"))

# Make a query
response = llm.query("What is the meaning of life?")
print(response)
```

## Configuration
YAMLLM uses YAML files for configuration. Set up a `.config` file to define the parameters for your LLM instance. This file should include settings such as the model type, temperature, maximum tokens, and system prompt.

Example configuration:

```yaml
  name: "openai"  # supported: openai, google, deepseek, mistral
  model: "gpt-4o-mini"  # model identifier
  api_key: # api key goes here, best practice to put into dotenv
  base_url: # optional: for custom endpoints

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
  system_prompt: "You are a helpful assistant, helping me achieve my goals"
  max_context_length: 16000
  memory:
    enabled: true
    max_messages: 10  # number of messages to keep in conversation history
    conversation_db: "yamllm/memory/conversation_history.db"
    vector_store:
      index_path: "yamllm/memory/vector_store/faiss_index.idx"
      metadata_path: "yamllm/memory/vector_store/metadata.pkl"
    
# Output Formatting
output:
  format: "text"  # supported: text, json, markdown
  stream: false

logging:
  level: "INFO"
  file: "yamllm.log"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Tool Management - In development
tools:
  enabled: false
  tool_timeout: 10  # seconds
  tool_list: ['calculator', 'web_search']

# Safety Settings
safety:
  content_filtering: true
  max_requests_per_minute: 60
  sensitive_keywords: []
```

Place the `.config` file in your project directory and reference it in your code to initialize the LLM instance.

## Features

- YAML-based configuration
- Simple API interface
- Customizable prompt templates
- Error handling and retry logic
- In built memory management in sqlite database

## License

MIT License