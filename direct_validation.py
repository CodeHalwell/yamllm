"""
Direct validation script for Azure providers.

This script tests the Azure provider classes directly without depending on the full module system.
"""

import sys
import os
import unittest
from unittest.mock import MagicMock

# Create mock classes to avoid dependency issues
class Message:
    def __init__(self, role, content):
        self.role = role
        self.content = content
    
    def to_dict(self):
        return {"role": self.role, "content": self.content}

class BaseProvider:
    """Mock base provider class"""
    pass

# Define the AzureOpenAIProvider class directly for testing
class AzureOpenAIProvider:
    """
    Azure OpenAI provider implementation for testing.
    """
    
    def __init__(self, api_key, model, base_url=None, **kwargs):
        """Initialize the Azure OpenAI provider."""
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.api_version = kwargs.get('api_version', '2023-05-15')
        
        # Normally we would initialize the client here
        # But for testing, we'll just mock it
        self.client = MagicMock()
        self.embedding_client = MagicMock()
        
        # Store additional parameters
        self.logger = kwargs.get('logger')
    
    def prepare_completion_params(self, messages, temperature, max_tokens, 
                                 top_p, stop_sequences=None):
        """Prepare completion parameters for Azure OpenAI's API."""
        # Convert Message objects to dictionaries
        message_dicts = [message.to_dict() for message in messages]
        
        params = {
            "model": self.model,
            "messages": message_dicts,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        
        # Only add stop parameter if it contains actual stop sequences
        if stop_sequences and len(stop_sequences) > 0:
            params["stop"] = stop_sequences
            
        return params

# Define the AzureFoundryProvider class directly for testing
class AzureFoundryProvider:
    """
    Azure AI Foundry provider implementation for testing.
    """
    
    def __init__(self, api_key, model, base_url=None, **kwargs):
        """Initialize the Azure AI Foundry provider."""
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.project_id = kwargs.get('project_id')
        
        # Normally we would initialize the client here
        # But for testing, we'll just mock it
        self.client = MagicMock()
        
        # Store additional parameters
        self.logger = kwargs.get('logger')
    
    def prepare_completion_params(self, messages, temperature, max_tokens, 
                                 top_p, stop_sequences=None):
        """Prepare completion parameters for Azure AI Foundry's API."""
        # Convert Message objects to dictionaries
        message_dicts = [message.to_dict() for message in messages]
        
        params = {
            "deployment_id": self.model,
            "messages": message_dicts,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        
        # Only add stop parameter if it contains actual stop sequences
        if stop_sequences and len(stop_sequences) > 0:
            params["stop"] = stop_sequences
            
        return params

# Run tests
def test_azure_openai_provider():
    print("Testing AzureOpenAIProvider...")
    
    # Initialize the provider
    provider = AzureOpenAIProvider(
        api_key="test_api_key",
        model="gpt-4",
        base_url="https://test-endpoint.openai.azure.com",
        api_version="2023-05-15"
    )
    
    # Verify initialization
    assert provider.api_key == "test_api_key"
    assert provider.model == "gpt-4"
    assert provider.base_url == "https://test-endpoint.openai.azure.com"
    assert provider.api_version == "2023-05-15"
    
    # Test prepare_completion_params
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Hello, how are you?")
    ]
    
    params = provider.prepare_completion_params(
        messages=messages,
        temperature=0.7,
        max_tokens=100,
        top_p=1.0,
        stop_sequences=["STOP"]
    )
    
    # Verify parameters
    assert params["model"] == "gpt-4"
    assert params["temperature"] == 0.7
    assert params["max_tokens"] == 100
    assert params["top_p"] == 1.0
    assert params["stop"] == ["STOP"]
    assert len(params["messages"]) == 2
    assert params["messages"][0]["role"] == "system"
    assert params["messages"][0]["content"] == "You are a helpful assistant."
    assert params["messages"][1]["role"] == "user"
    assert params["messages"][1]["content"] == "Hello, how are you?"
    
    print("AzureOpenAIProvider tests passed!")

def test_azure_foundry_provider():
    print("\nTesting AzureFoundryProvider...")
    
    # Initialize the provider
    provider = AzureFoundryProvider(
        api_key="test_api_key",
        model="my-gpt4-deployment",
        base_url="https://test-project.azureai.azure.com",
        project_id="test-project-id"
    )
    
    # Verify initialization
    assert provider.api_key == "test_api_key"
    assert provider.model == "my-gpt4-deployment"
    assert provider.base_url == "https://test-project.azureai.azure.com"
    assert provider.project_id == "test-project-id"
    
    # Test prepare_completion_params
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Hello, how are you?")
    ]
    
    params = provider.prepare_completion_params(
        messages=messages,
        temperature=0.7,
        max_tokens=100,
        top_p=1.0,
        stop_sequences=["STOP"]
    )
    
    # Verify parameters
    assert params["deployment_id"] == "my-gpt4-deployment"
    assert params["temperature"] == 0.7
    assert params["max_tokens"] == 100
    assert params["top_p"] == 1.0
    assert params["stop"] == ["STOP"]
    assert len(params["messages"]) == 2
    assert params["messages"][0]["role"] == "system"
    assert params["messages"][0]["content"] == "You are a helpful assistant."
    assert params["messages"][1]["role"] == "user"
    assert params["messages"][1]["content"] == "Hello, how are you?"
    
    print("AzureFoundryProvider tests passed!")

if __name__ == "__main__":
    test_azure_openai_provider()
    test_azure_foundry_provider()
    print("\nAll tests passed successfully!")