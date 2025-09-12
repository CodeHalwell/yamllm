# Azure Semantic Kernel Integration Approach

This document outlines an approach for integrating [Azure Semantic Kernel](https://learn.microsoft.com/en-us/semantic-kernel/overview/) with yamllm-core.

## Overview

Azure Semantic Kernel is a framework that orchestrates AI models and plugins to create AI applications. It provides a way to combine language models with traditional code to create AI experiences. 

Semantic Kernel could be integrated with yamllm-core in multiple ways:

1. **Provider Pattern**: Implement a `SemanticKernelProvider` that uses Semantic Kernel as a backend for LLM interactions
2. **Orchestration Layer**: Use Semantic Kernel as a higher-level orchestration layer on top of yamllm-core

## 1. Provider Pattern Implementation

### Configuration

```yaml
provider:
  name: "semantic_kernel"
  model: "gpt-4"  # Model used by Semantic Kernel
  # Do not include API keys in config. Pass via environment and constructor.
  base_url: ${AZURE_OPENAI_ENDPOINT}  # For Azure OpenAI backend
  extra_settings:
    api_version: "2023-05-15"  # For Azure OpenAI
    kernel_plugins: ["web-search", "math", "time"]  # Plugins to load
    planner: "sequential"  # Planning strategy
```

### Implementation

The provider would need to:

1. Initialize a Semantic Kernel instance
2. Load configured plugins
3. Set up the planner
4. Route chat completions through the kernel

Example implementation skeleton:

```python
from typing import Dict, List, Any, Optional
import json
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from yamllm.core.providers.base import BaseProvider

class SemanticKernelProvider(BaseProvider):
    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None, **kwargs):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        
        # Extract Semantic Kernel-specific parameters
        self.api_version = kwargs.get('api_version', '2023-05-15')
        self.kernel_plugins = kwargs.get('kernel_plugins', [])
        self.planner_type = kwargs.get('planner', 'sequential')
        
        # Initialize Semantic Kernel
        self.kernel = sk.Kernel()
        
        # Add AI service
        self.kernel.add_chat_service(
            "azure_chat_completion", 
            AzureChatCompletion(
                deployment_name=self.model,
                endpoint=self.base_url,
                api_key=self.api_key,
                api_version=self.api_version
            )
        )
        
        # Load plugins
        self._load_plugins()
        
        # Initialize planner
        self._setup_planner()
        
    def _load_plugins(self):
        # Load plugins based on configuration
        for plugin_name in self.kernel_plugins:
            # Implementation depends on plugin type
            pass
            
    def _setup_planner(self):
        # Setup planner based on configuration
        if self.planner_type == "sequential":
            self.planner = sk.planning.SequentialPlanner(self.kernel)
        elif self.planner_type == "action":
            self.planner = sk.planning.ActionPlanner(self.kernel)
        else:
            raise ValueError(f"Unknown planner type: {self.planner_type}")
            
    # Implement other BaseProvider methods...
```

## 2. Orchestration Layer Approach

In this approach, Semantic Kernel would be used as a higher-level orchestration layer that can use yamllm-core as one of its services.

### Architecture

1. Semantic Kernel initializes and manages the application flow
2. yamllm-core instances are registered as services or plugins within the kernel
3. The kernel plans and executes workflows that may involve multiple LLM calls

### Implementation Example

```python
import semantic_kernel as sk
from yamllm.core.llm import LLM

# Initialize Semantic Kernel
kernel = sk.Kernel()

# Initialize yamllm-core instance
llm = LLM(config_path="config.yaml", api_key="your-api-key")

# Register yamllm-core as a plugin
@sk.kernel_function
def ask_yamllm(question: str) -> str:
    """Ask a question to yamllm-core"""
    return llm.query(question)

# Add the function to the kernel
yamllm_plugin = kernel.create_function_from_method(ask_yamllm)
kernel.add_plugin("yamllm", {"ask": yamllm_plugin})

# Use in a plan
plan = kernel.create_semantic_function("""
    To answer this question, I need to:
    1. Ask yamllm about {{$input}}
    2. Analyze the response
    3. Provide a clear answer
    
    First, let's ask yamllm: {{yamllm.ask $input}}
    
    Based on this information, my answer is:
""")

result = plan.invoke("What is the capital of France?")
print(result)
```

## 3. Hybrid Approach

A hybrid approach could also be implemented where yamllm-core can optionally use Semantic Kernel for certain complex tasks:

```yaml
provider:
  name: "openai"  # Regular provider
  model: "gpt-4"
  api_key: ${OPENAI_API_KEY}
  
tools:
  enabled: true
  tool_list: ["semantic_kernel_planner"]
  semantic_kernel:
    enabled: true
    plugins: ["web-search", "math", "time"]
    planner: "sequential"
```

## Benefits and Challenges

### Benefits

1. **Planning Capabilities**: Semantic Kernel provides sophisticated planning capabilities
2. **Plugin Ecosystem**: Access to Semantic Kernel's growing plugin ecosystem
3. **Memory Systems**: Built-in conversational and semantic memory systems
4. **Integration with Microsoft Stack**: Better integration with Azure services

### Challenges

1. **Complexity**: Adds significant complexity to the architecture
2. **Overlap in Functionality**: Some features like tool calling overlap with yamllm-core
3. **Development Overhead**: Requires maintaining compatibility with Semantic Kernel versions
4. **Paradigm Differences**: Semantic Kernel's programming model differs from yamllm-core's YAML-first approach

## Recommendation

For most use cases, the Provider Pattern (Option 1) offers the best balance of functionality and integration complexity. It allows yamllm-core users to benefit from Semantic Kernel capabilities while maintaining the familiar YAML configuration approach.

The full Orchestration Layer approach (Option 2) would be better suited for applications that are primarily built around Semantic Kernel and only need occasional yamllm-core functionality.

## Implementation Priority

1. Implement basic `AzureOpenAIProvider` for direct access to Azure OpenAI models
2. Implement `AzureFoundryProvider` for access to models in Azure AI Foundry
3. Document the Semantic Kernel integration approach for future implementation
4. Consider implementing the Provider Pattern for Semantic Kernel if there is user demand
