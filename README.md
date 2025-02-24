# YAMLLM

A Python library for YAML-based LLM configuration and execution.

## Installation

```bash
pip install yamllm-core
```

```bash
uv add yamllm-core
```

## Quick Start

In order to run a simple query, run a script as follows. NOTE: Printing of the response is not required as this is handles by the query method. This uses the rich library to print the responses in the console.

```python
from yamllm import OpenAIGPT, GoogleGemini, DeepSeek, MistralAI
import os
import dotenv

dotenv.load_dotenv()

config_path = "path/to/config.yaml"

# Initialize LLM with config
llm = GoogleGemini(config_path=config_path, api_key=os.environ.get("GOOGLE_API_KEY"))

# Make a query
response = llm.query("Give me some boiler plate pytorch code please")
```

In order to have an ongoing conversation with the model, run a script as follows.

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

## Working with Conversation History

You can view your conversation history using the ConversationStore class. This will display all messages in a tabulated format:

```python
from yamllm import ConversationStore
import pandas as pd
from tabulate import tabulate

# Initialize the conversation store
history = ConversationStore("yamllm/memory/conversation_history.db")

# Retrieve messages and create a DataFrame
messages = history.get_messages()
df = pd.DataFrame(messages)

# Display messages in tabular format
print(tabulate(df, headers='keys', tablefmt='psql'))
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

## Configuration
YAMLLM uses YAML files for configuration. Set up a `.config` file to define the parameters for your LLM instance. This file should include settings such as the model type, temperature, maximum tokens, and system prompt.

Example configuration:

```yaml
  name: "openai"  # supported: openai, google, deepseek, mistral
  model: "gpt-4o-mini"  # model identifier
  api_key: # api key goes here, best practice to put into dotenv
  base_url: # optional: for custom endpoints e.g. "https://generativelanguage.googleapis.com/v1beta/openai/"

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
- In built memory management in sqlite database for short term memory
- Use of vector database for long term memory based on semantic search
- Choose streaming or non-streamed response

## License

MIT License