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

## Mistral and DeepSeek

These providers are also supported through their OpenAI-compatible endpoints.

```python
from yamllm.core.llm import MistralAI, DeepSeek
# Or use the new interface
from yamllm.core.llm_v2 import LLMv2

# Using original classes
llm = MistralAI(config_path="config.yaml", api_key="your-api-key")
# or
llm = DeepSeek(config_path="config.yaml", api_key="your-api-key")

# Using new provider interface
llm = LLMv2(config_path="config.yaml", api_key="your-api-key")
```

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