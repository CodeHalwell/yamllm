
# Usage Guide

## Basic Usage

```python
from yamllm.core.llm import GoogleGemini, DeepSeek, OpenAIGPT, Mistral
import dotenv
import os

dotenv.load_dotenv()

# Initialize LLM with config
llm = Mistral(config_path="path/to/config.yaml", api_key=os.environ.get("MISTRAL_API_KEY"))

# Make a query
response = llm.query("Give me some boiler plate pytorch code please")
```

## Advanced Usage

```python
from yamllm.core.llm import GoogleGemini, DeepSeek, OpenAIGPT, Mistral
import dotenv
import os
from rich.console import Console

# Initialize pretty printer
console = Console()
dotenv.load_dotenv()

# Get the absolute path to the config file
config_path = "mistral_config.yaml"

llm = MistralAI(config_path=config_path, api_key=os.environ.get("MISTRAL_API_KEY"))

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