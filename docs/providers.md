# Provider Interface

YAMLLM now supports a provider-agnostic interface that allows using different LLM providers with a unified API. The key providers currently supported are:

## OpenAI (GPT Models)

```python
from yamllm.core.llm import OpenAIGPT
# Or use the new interface
from yamllm.core.llm_v2 import LLMv2

# Using original class
llm = OpenAIGPT(config_path="config.yaml", api_key="your-api-key")

# Using new provider interface
llm = LLMv2(config_path="config.yaml", api_key="your-api-key")
```

Configuration:
```yaml
provider:
  name: "openai"
  model: "gpt-4o-mini"  # model identifier
  api_key: # api key goes here, best practice to put into dotenv
  base_url: # optional: for custom endpoints
```

**Tool Support:** Full support for function calling using OpenAI's native tool use capabilities. Compatible with the latest models including GPT-4o.

## Anthropic (Claude Models)

```python
from yamllm.core.llm import AnthropicAI
# Or use the new interface
from yamllm.core.llm_v2 import LLMv2

# Using original class
llm = AnthropicAI(config_path="config.yaml", api_key="your-api-key")

# Using new provider interface
llm = LLMv2(config_path="config.yaml", api_key="your-api-key")
```

Configuration:
```yaml
provider:
  name: "anthropic"
  model: "claude-3-opus-20240229"  # model identifier
  api_key: # api key goes here, best practice to put into dotenv
  base_url: # optional: for custom endpoints
  extra_settings:
    api_version: "2023-06-01"  # Anthropic API version
```

**Tool Support:** Full support for tool use with Claude 3 models using Anthropic's native tool use capabilities. The implementation uses Anthropic's latest SDK pattern for tool calling.

## Google (Gemini Models)

```python
from yamllm.core.llm import GoogleGemini
# Or use the new interface
from yamllm.core.llm_v2 import LLMv2

# Using original class
llm = GoogleGemini(config_path="config.yaml", api_key="your-api-key")

# Using new provider interface
llm = LLMv2(config_path="config.yaml", api_key="your-api-key")
```

Configuration:
```yaml
provider:
  name: "google"
  model: "gemini-1.5-flash"  # model identifier
  api_key: # api key goes here, best practice to put into dotenv
  base_url: null  # optional: for custom endpoints, e.g. "https://generativelanguage.googleapis.com/v1"
```

The GoogleGeminiProvider now uses the native Google GenAI SDK for improved performance, better access to Gemini-specific features (especially tool use), and alignment with Google's recommended practices.

**Tool Support:** Full support for function calling using Google's native function declarations format. The implementation converts YAMLLM's standardized tool definitions to Google's format and properly handles the function_response objects.

## Mistral 

```python
from yamllm.core.llm import MistralAI
# Or use the new interface
from yamllm.core.llm_v2 import LLMv2

# Using original class
llm = MistralAI(config_path="config.yaml", api_key="your-api-key")

# Using new provider interface
llm = LLMv2(config_path="config.yaml", api_key="your-api-key")
```

Configuration:
```yaml
provider:
  name: "mistralai"
  model: "mistral-small-latest"  # model identifier
  api_key: # api key goes here, best practice to put into dotenv
  base_url: "https://api.mistral.ai/v1/" # optional: for custom endpoints
```

The MistralProvider uses the official mistralai Python SDK for improved performance, better access to Mistral-specific features (like the mistral-embed model for embeddings), and robust tool usage support.

**Tool Support:** Full support for tool use with Mistral models that support function calling. Mistral uses an OpenAI-compatible format for function calling.

## DeepSeek

DeepSeek is supported through an OpenAI-compatible endpoint with some specific optimizations.

```python
from yamllm.core.llm import DeepSeek
# Or use the new interface
from yamllm.core.llm_v2 import LLMv2

# Using original classes
llm = DeepSeek(config_path="config.yaml", api_key="your-api-key")

# Using new provider interface
llm = LLMv2(config_path="config.yaml", api_key="your-api-key")
```

Configuration:
```yaml
provider:
  name: "deepseek"
  model: "deepseek-chat"  # model identifier
  api_key: # api key goes here, best practice to put into dotenv
  base_url: "https://api.deepseek.com" # optional: for custom endpoints
  extra_settings:
    headers:
      User-Agent: "YAMLLM/1.0"  # optional: custom user agent
    cache_enabled: true         # optional: enable request caching if supported
    cache_ttl: 3600             # optional: time-to-live for cached requests (seconds)
```

**Note on Embeddings**: DeepSeek may not support embeddings through their OpenAI-compatible API, or may use different embedding models. The current implementation falls back to OpenAI's embedding model, which may not be optimal for DeepSeek.

**Tool Support:** Support for function calling using DeepSeek's OpenAI-compatible API. Since DeepSeek uses an OpenAI-compatible API, it inherits the OpenAI provider's tool use implementation.

## Azure OpenAI

Azure OpenAI is supported through the Azure OpenAI Service.

```python
from yamllm.core.llm import AzureOpenAI
# Or use the new interface
from yamllm.core.llm_v2 import LLMv2

# Using original classes
llm = AzureOpenAI(config_path="config.yaml", api_key="your-api-key")

# Using new provider interface
llm = LLMv2(config_path="config.yaml", api_key="your-api-key")
```

Configuration:
```yaml
provider:
  name: "azure_openai"
  model: "your-deployment-name"  # deployment name in Azure
  api_key: # api key goes here, best practice to put into dotenv
  base_url: "https://your-resource-name.openai.azure.com/" # Azure endpoint
  extra_settings:
    api_version: "2023-05-15"  # Azure OpenAI API version
    embedding_deployment: "text-embedding-ada-002"  # optional: deployment for embeddings
```

**Tool Support:** Full support for function calling using Azure OpenAI's native tool use capabilities, which are compatible with OpenAI's format.

## Azure AI Foundry

Azure AI Foundry is supported for access to custom models deployed in Azure AI Foundry projects.

```python
from yamllm.core.llm import AzureFoundry
# Or use the new interface
from yamllm.core.llm_v2 import LLMv2

# Using original classes
llm = AzureFoundry(config_path="config.yaml", api_key="your-api-key")

# Using new provider interface
llm = LLMv2(config_path="config.yaml", api_key="your-api-key")
```

Configuration:
```yaml
provider:
  name: "azure_foundry"
  model: "your-deployment-name"  # deployment name in Azure AI Foundry
  api_key: "your-api-key" # or "default" to use DefaultAzureCredential
  base_url: "https://your-project-endpoint.ai.azure.com" # Azure AI project endpoint
  extra_settings:
    project_id: "your-project-id"  # optional if included in endpoint
    embedding_deployment: "text-embedding-ada-002"  # optional: deployment for embeddings
```

**Tool Support:** Support for function calling using Azure AI Foundry's tool use capabilities, which follow a format similar to OpenAI's API.

## Provider Interface

For developers who want to add new providers, the provider interface can be extended:

```python
from yamllm.core.providers.base import BaseProvider

class MyCustomProvider(BaseProvider):
    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        # Initialize your provider
        pass
        
    def get_completion(self, messages, model, temperature, max_tokens, top_p, stop_sequences=None, tools=None, stream=False, **kwargs):
        # Implement completion method
        pass
        
    def get_streaming_completion(self, messages, model, temperature, max_tokens, top_p, stop_sequences=None, tools=None, **kwargs):
        # Implement streaming completion method
        pass
        
    def create_embedding(self, text, model):
        # Implement embedding creation
        pass
        
    def format_tool_calls(self, tool_calls):
        # Format provider-specific tool calls to standard format
        pass
        
    def format_tool_results(self, tool_results):
        # Format standard tool results to provider-specific format
        pass
        
    def close(self):
        # Close any resources
        pass
```

Then register your provider in the `PROVIDER_MAP` in `LLMv2` class.

## Tool Use Configuration

To enable tools in your YAML configuration:

```yaml
tools:
  enabled: true                    # Enable tool use
  tool_timeout: 30                 # Maximum time in seconds for tool execution
  tool_list:                       # List of tools to enable
    - "web_search"                 # DuckDuckGo web search
    - "calculator"                 # Evaluate mathematical expressions
    - "timezone"                   # Convert between timezones
    - "unit_converter"             # Convert between units
    - "weather"                    # Get weather information
    - "web_scraper"                # Scrape content from websites
```

All supported providers implement a standardized interface for tool use, ensuring consistent behavior across different LLM backends.