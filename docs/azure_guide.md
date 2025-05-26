# Azure AI Integration Guide for yamllm-core

This guide provides information on how to use the Azure AI integrations in yamllm-core.

## Available Azure Integrations

yamllm-core now supports the following Azure AI integrations:

1. **Azure OpenAI**: Direct integration with Azure OpenAI Service
2. **Azure AI Foundry**: Integration with Azure AI Foundry for model deployments

## Azure OpenAI Integration

### Configuration

To use Azure OpenAI with yamllm-core, create a configuration file like this:

```yaml
provider:
  name: "azure_openai"
  model: "gpt-4"  # Your deployed model name in Azure OpenAI
  api_key: ${AZURE_OPENAI_API_KEY}  # Use environment variable for API key
  base_url: ${AZURE_OPENAI_ENDPOINT}  # Azure OpenAI endpoint
  extra_settings:
    api_version: "2023-05-15"  # Azure OpenAI API version
    embedding_deployment: "text-embedding-ada-002"  # Optional: specify embedding model
```

### Usage

```python
from yamllm.core.llm import LLM

# Initialize with Azure OpenAI configuration
llm = LLM(config_path="azure_openai_config.yaml", api_key="your-api-key")

# Use like any other yamllm LLM
response = llm.query("What is the capital of France?")
print(response)
```

## Azure AI Foundry Integration

### Configuration

To use Azure AI Foundry with yamllm-core, create a configuration file like this:

```yaml
provider:
  name: "azure_foundry"
  model: "my-gpt4-deployment"  # Your model deployment name in Azure AI Foundry
  api_key: "default"  # Use "default" for DefaultAzureCredential or your API key
  base_url: ${AZURE_AI_PROJECT_ENDPOINT}  # Azure AI Foundry project endpoint
  extra_settings:
    project_id: "my-project-id"  # Optional: your Azure AI project ID
    embedding_deployment: "text-embedding-ada-002"  # Optional: specify embedding model
```

### Usage

```python
from yamllm.core.llm import LLM

# Initialize with Azure AI Foundry configuration
llm = LLM(config_path="azure_foundry_config.yaml", api_key="your-api-key")

# Use like any other yamllm LLM
response = llm.query("What is the capital of France?")
print(response)
```

## Authentication Options

Azure integrations support two authentication methods:

1. **API Key**: Provide an API key for authentication
   ```yaml
   provider:
     api_key: "your-api-key-here"
   ```

2. **DefaultAzureCredential**: Use the Azure Identity library for authentication
   ```yaml
   provider:
     api_key: "default"  # Special value that triggers DefaultAzureCredential
   ```

   This method will attempt to authenticate using environment variables, managed identity, Visual Studio Code credentials, Azure CLI credentials, and more.

## Advanced Integrations

For more advanced Azure AI integrations, see:

- [Azure Semantic Kernel Integration](/docs/integrations/azure_semantic_kernel.md)
- [Azure AI Agent Integration](/docs/integrations/azure_ai_agent.md)

These documents outline theoretical approaches for integrating with Azure Semantic Kernel and Azure AI Agent Service, which may be implemented in future versions.

## Troubleshooting

### Common Issues

1. **Authentication Errors**:
   - Ensure your API key is correct
   - If using DefaultAzureCredential, ensure you're properly authenticated to Azure

2. **Endpoint Issues**:
   - Verify your Azure OpenAI or AI Foundry endpoint URL
   - Ensure your resource is provisioned and available

3. **Model Deployment**:
   - Verify that the model name matches your deployment name in Azure
   - Check that your model deployment is active

4. **API Version**:
   - Ensure you're using a compatible API version for Azure OpenAI

### Getting Help

If you encounter issues with Azure integrations, please:

1. Check the detailed error message for clues
2. Verify your Azure resources are correctly configured
3. Consult the Azure OpenAI or Azure AI Foundry documentation
4. Open an issue on the yamllm GitHub repository with detailed information about your problem