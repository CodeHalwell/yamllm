# Azure AI Agent Service Integration Approach

This document outlines an approach for integrating [Azure AI Agent Service](https://learn.microsoft.com/en-us/azure/ai-services/azure-ai-agent-service/overview) with yamllm-core.

## Overview

Azure AI Agent Service is a fully managed service for building, deploying, and running AI agents in the cloud. It provides features like:

- Agent orchestration
- Tool integration
- Function calling
- File handling
- Code interpretation
- Memory and state management

Azure AI Agent Service could be integrated with yamllm-core in multiple ways:

1. **Provider Pattern**: Implement an `AzureAIAgentProvider` that uses Azure AI Agent Service as a backend
2. **Tool Integration**: Use Azure AI Agent Service as a tool within yamllm-core
3. **Hybrid Approach**: Create a bridge between yamllm-core and Azure AI Agent Service

## 1. Provider Pattern Implementation

### Configuration

```yaml
provider:
  name: "azure_ai_agent"
  model: "gpt-4"  # Base model for the agent
  # Do not include API keys in config. Pass via environment and constructor.
  base_url: ${AZURE_AGENT_ENDPOINT}
  extra_settings:
    project_id: "my-project-id"
    agent_id: "my-agent-id"
    tools: ["web-search", "calculator", "weather"]
```

### Implementation

The provider would need to:

1. Initialize the Azure AI Agent client
2. Configure the agent with tools and settings
3. Route completions through the agent

Example implementation skeleton:

```python
from typing import Dict, List, Any, Optional
import json
from azure.ai.agents import AgentClient
from azure.identity import DefaultAzureCredential
from yamllm.core.providers.base import BaseProvider

class AzureAIAgentProvider(BaseProvider):
    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None, **kwargs):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        
        # Extract Azure AI Agent-specific parameters
        self.project_id = kwargs.get('project_id')
        self.agent_id = kwargs.get('agent_id')
        self.tools = kwargs.get('tools', [])
        
        # Initialize Azure AI Agent client
        if self.api_key.lower() == "default":
            credential = DefaultAzureCredential()
            self.client = AgentClient(endpoint=self.base_url, credential=credential)
        else:
            self.client = AgentClient(endpoint=self.base_url, api_key=self.api_key)
        
        # Store additional parameters
        self.logger = kwargs.get('logger')
    
    def prepare_completion_params(self, messages: List[Message], temperature: float, max_tokens: int, 
                                 top_p: float, stop_sequences: Optional[List[str]] = None) -> Dict[str, Any]:
        # Convert Message objects to agent-compatible format
        agent_messages = [{"role": message.role, "content": message.content} for message in messages]
        
        params = {
            "project_id": self.project_id,
            "agent_id": self.agent_id,
            "messages": agent_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        
        if stop_sequences and len(stop_sequences) > 0:
            params["stop"] = stop_sequences
            
        return params
    
    # Implement other BaseProvider methods...
```

## 2. Tool Integration Approach

In this approach, an Azure AI Agent is used as a tool within yamllm-core:

### Configuration

```yaml
provider:
  name: "openai"  # Regular provider
  model: "gpt-4"
  # Do not include API keys in config. Pass via environment and constructor.
  
tools:
  enabled: true
  tool_list: ["azure_agent"]
  azure_agent:
    enabled: true
    endpoint: ${AZURE_AGENT_ENDPOINT}
    project_id: "my-project-id"
    agent_id: "my-agent-id"
```

### Implementation Example

```python
from azure.ai.agents import AgentClient
from azure.identity import DefaultAzureCredential

class AzureAgentTool:
    def __init__(self, endpoint, project_id, agent_id, api_key=None):
        self.endpoint = endpoint
        self.project_id = project_id
        self.agent_id = agent_id
        
        # Initialize Azure AI Agent client
        if api_key:
            self.client = AgentClient(endpoint=endpoint, api_key=api_key)
        else:
            credential = DefaultAzureCredential()
            self.client = AgentClient(endpoint=endpoint, credential=credential)
    
    def execute(self, query: str):
        """
        Execute a query using Azure AI Agent.
        
        Args:
            query (str): The query to send to the agent.
            
        Returns:
            str: The response from the agent.
        """
        response = self.client.create_conversation(
            project_id=self.project_id,
            agent_id=self.agent_id,
            messages=[{"role": "user", "content": query}]
        )
        
        return response.response.content
```

## 3. Hybrid Approach

A hybrid approach would integrate yamllm-core with Azure AI Agent Service at a deeper level, allowing for more complex interactions.

In this approach, yamllm-core would be extended with a new component that can:

1. Create and manage Azure AI Agents
2. Configure agents with yamllm-core tools
3. Bridge between local tools and cloud-based agents

```python
from yamllm.core.llm import LLM
from azure.ai.agents import AgentClient, AgentConfiguration, ToolConfiguration

class AzureAgentManager:
    def __init__(self, config_path, endpoint, project_id=None):
        self.llm = LLM(config_path=config_path)
        self.endpoint = endpoint
        self.project_id = project_id
        self.client = AgentClient(endpoint=endpoint)
        
    def create_agent_from_tools(self, name, description, tools):
        """
        Create an Azure AI Agent with the specified tools.
        
        Args:
            name (str): The name of the agent.
            description (str): The description of the agent.
            tools (List[str]): The tools to enable for the agent.
            
        Returns:
            str: The agent ID.
        """
        # Get tools from yamllm-core and convert to agent tool configuration
        tool_configs = []
        for tool_name in tools:
            if tool_name in self.llm.available_tools:
                tool = self.llm.available_tools[tool_name]
                tool_config = ToolConfiguration(
                    name=tool_name,
                    description=tool.__doc__,
                    # Map yamllm tool schema to Azure AI Agent tool schema
                    # (implementation details would vary)
                )
                tool_configs.append(tool_config)
        
        # Create agent configuration
        agent_config = AgentConfiguration(
            name=name,
            description=description,
            tools=tool_configs
        )
        
        # Create the agent
        response = self.client.create_agent(
            project_id=self.project_id,
            agent_configuration=agent_config
        )
        
        return response.agent_id
```

## Benefits and Challenges

### Benefits

1. **Managed Infrastructure**: Azure AI Agent Service provides fully managed infrastructure
2. **Advanced Features**: Access to advanced features like file handling and code interpretation
3. **Integration with Azure Ecosystem**: Better integration with Azure services
4. **Security and Compliance**: Enterprise-grade security and compliance

### Challenges

1. **Vendor Lock-in**: Deeper integration with Azure may limit portability
2. **Latency**: Network calls to Azure may introduce latency
3. **Cost**: Azure AI Agent Service may incur additional costs
4. **Complexity**: Integration adds complexity to the architecture

## Recommendation

For most use cases, the Provider Pattern (Option 1) offers the best balance of functionality and integration complexity. It allows yamllm-core users to benefit from Azure AI Agent Service capabilities while maintaining the familiar YAML configuration approach.

The Tool Integration approach (Option 2) would be better suited for applications that only need occasional access to Azure AI Agent Service.

## Implementation Priority

1. Implement basic `AzureOpenAIProvider` for direct access to Azure OpenAI models
2. Implement `AzureFoundryProvider` for access to models in Azure AI Foundry
3. Document the Azure AI Agent Service integration approach for future implementation
4. Consider implementing the Provider Pattern for Azure AI Agent Service if there is user demand
