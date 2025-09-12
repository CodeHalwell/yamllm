# Azure AI Integration

yamllm-core now supports integration with various Azure AI services, providing access to Azure-hosted models and capabilities.

## Available Azure Integrations

### Azure OpenAI

Direct integration with Azure OpenAI Service, allowing you to use your Azure-hosted OpenAI models.

```python
from yamllm.core.llm import LLM

# Using Azure OpenAI
llm = LLM(config_path="azure_openai_config.yaml", api_key="your-api-key")
```

Configuration:
```yaml
provider:
  name: "azure_openai"
  model: "gpt-4"  # your deployed model name
  # Do not include API keys in config. Pass via environment and constructor.
  base_url: ${AZURE_OPENAI_ENDPOINT}
  extra_settings:
    api_version: "2023-05-15"  # Azure OpenAI API version
```

### Azure AI Foundry

Integration with Azure AI Foundry, allowing you to use models deployed in your Azure AI projects.

```python
from yamllm.core.llm import LLM

# Using Azure AI Foundry
llm = LLM(config_path="azure_foundry_config.yaml", api_key="your-api-key")
```

Configuration:
```yaml
provider:
  name: "azure_foundry"
  model: "my-gpt4-deployment"  # your model deployment name
  # For DefaultAzureCredential, pass api_key="default" to your constructor at runtime
  base_url: ${AZURE_AI_PROJECT_ENDPOINT}
  extra_settings:
    project_id: "my-project-id"  # optional: your Azure AI project ID
```

## Authentication

Azure integrations support two authentication methods:

1. **API Key**: Provide an API key for authentication
   ```yaml
   provider:
     # Do not include credentials in YAML
   ```

2. **DefaultAzureCredential**: Use the Azure Identity library for authentication
   ```yaml
   provider:
     # At runtime, pass api_key="default" to use DefaultAzureCredential
   ```

   This method will attempt to authenticate using environment variables, managed identity, Visual Studio Code credentials, Azure CLI credentials, and more.

## Future Integrations

We are exploring more advanced integrations with Azure AI services:

1. **Azure Semantic Kernel**: Integration with the Semantic Kernel framework for advanced planning and orchestration
2. **Azure AI Agent Service**: Integration with the Azure AI Agent Service for managed agent capabilities

For more information on these future integrations, see:
- [Azure Semantic Kernel Integration](/docs/integrations/azure_semantic_kernel.md)
- [Azure AI Agent Integration](/docs/integrations/azure_ai_agent.md)
