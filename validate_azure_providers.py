"""
Simple validation test for Azure providers.

This script performs a minimal validation of the Azure provider classes
without requiring all dependencies.
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
import json
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Create mock classes to avoid dependency issues
class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content
    
    def to_dict(self):
        return {"role": self.role, "content": self.content}

# Override the imports in the azure_openai_provider module
sys.modules['openai'] = MagicMock()
sys.modules['rich.console'] = MagicMock()
sys.modules['rich.markdown'] = MagicMock()
sys.modules['rich.live'] = MagicMock()
sys.modules['yamllm.providers.base'] = MagicMock()
sys.modules['yamllm.providers.base'].Message = MockMessage

# Import the module after mocking dependencies
with patch('yamllm.providers.azure_openai_provider.AzureOpenAI'):
    from yamllm.providers.azure_openai_provider import AzureOpenAIProvider

print("Successfully imported AzureOpenAIProvider!")

# Test initialization
provider = AzureOpenAIProvider(
    api_key="test_key",
    model="gpt-4",
    base_url="https://test.openai.azure.com",
    api_version="2023-05-15"
)

print("Successfully initialized AzureOpenAIProvider!")

# Test preparation of completion parameters
messages = [MockMessage("system", "You are a helpful assistant."), 
            MockMessage("user", "Hello!")]

params = provider.prepare_completion_params(
    messages=messages,
    temperature=0.7,
    max_tokens=100,
    top_p=1.0,
    stop_sequences=["STOP"]
)

assert params["model"] == "gpt-4"
assert params["temperature"] == 0.7
assert params["max_tokens"] == 100
assert params["top_p"] == 1.0
assert params["stop"] == ["STOP"]
assert len(params["messages"]) == 2
assert params["messages"][0]["role"] == "system"
assert params["messages"][0]["content"] == "You are a helpful assistant."
assert params["messages"][1]["role"] == "user"
assert params["messages"][1]["content"] == "Hello!"

print("Successfully tested prepare_completion_params!")
print("All basic validation tests passed!")

print("\nValidating Azure Foundry Provider...")

# Override the imports in the azure_foundry_provider module
sys.modules['azure.ai.inference'] = MagicMock()
sys.modules['azure.identity'] = MagicMock()

# Import the module after mocking dependencies
with patch('yamllm.providers.azure_foundry_provider.InferenceClient'), \
     patch('yamllm.providers.azure_foundry_provider.DefaultAzureCredential'):
    from yamllm.providers.azure_foundry_provider import AzureFoundryProvider

print("Successfully imported AzureFoundryProvider!")

# Test initialization
foundry_provider = AzureFoundryProvider(
    api_key="test_key",
    model="my-gpt4-deployment",
    base_url="https://test-project.azureai.azure.com",
    project_id="test-project-id"
)

print("Successfully initialized AzureFoundryProvider!")

# Test preparation of completion parameters
params = foundry_provider.prepare_completion_params(
    messages=messages,
    temperature=0.7,
    max_tokens=100,
    top_p=1.0,
    stop_sequences=["STOP"]
)

assert params["deployment_id"] == "my-gpt4-deployment"
assert params["temperature"] == 0.7
assert params["max_tokens"] == 100
assert params["top_p"] == 1.0
assert params["stop"] == ["STOP"]
assert len(params["messages"]) == 2
assert params["messages"][0]["role"] == "system"
assert params["messages"][0]["content"] == "You are a helpful assistant."
assert params["messages"][1]["role"] == "user"
assert params["messages"][1]["content"] == "Hello!"

print("Successfully tested prepare_completion_params for AzureFoundryProvider!")
print("All basic validation tests passed!")