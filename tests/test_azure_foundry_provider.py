"""
Tests for Azure AI Foundry provider.

This module contains tests for the Azure AI Foundry provider implementation.
"""

import os
import unittest
from unittest.mock import MagicMock, patch
import sys
import json

from yamllm.providers.azure_foundry_provider import AzureFoundryProvider
from yamllm.providers.base import Message


class TestAzureFoundryProvider(unittest.TestCase):
    """Test cases for AzureFoundryProvider."""

    def setUp(self):
        """Set up test fixtures."""
        self.api_key = "test_api_key"
        self.model = "my-gpt4-deployment"
        self.base_url = "https://test-project.azureai.azure.com"
        self.project_id = "test-project-id"
        
        # Create a mock for the InferenceClient
        self.mock_client_patcher = patch('yamllm.providers.azure_foundry_provider.InferenceClient')
        self.mock_client = self.mock_client_patcher.start()
        
        # Create a mock for DefaultAzureCredential
        self.mock_credential_patcher = patch('yamllm.providers.azure_foundry_provider.DefaultAzureCredential')
        self.mock_credential = self.mock_credential_patcher.start()
        
        # Set up the provider
        self.provider = AzureFoundryProvider(
            api_key=self.api_key,
            model=self.model,
            base_url=self.base_url,
            project_id=self.project_id
        )

    def tearDown(self):
        """Tear down test fixtures."""
        self.mock_client_patcher.stop()
        self.mock_credential_patcher.stop()

    def test_init(self):
        """Test initialization of the provider."""
        # Assert that the provider was initialized correctly
        self.assertEqual(self.provider.api_key, self.api_key)
        self.assertEqual(self.provider.model, self.model)
        self.assertEqual(self.provider.base_url, self.base_url)
        self.assertEqual(self.provider.project_id, self.project_id)
        
        # Assert that the InferenceClient was initialized with the correct parameters
        self.mock_client.assert_called_with(
            endpoint=self.base_url,
            api_key=self.api_key
        )

    def test_init_with_default_credential(self):
        """Test initialization with DefaultAzureCredential."""
        # Create provider with 'default' api_key
        provider = AzureFoundryProvider(
            api_key="default",
            model=self.model,
            base_url=self.base_url
        )
        
        # Assert that DefaultAzureCredential was created
        self.mock_credential.assert_called_once()
        
        # Assert that InferenceClient was initialized with the credential
        self.mock_client.assert_called_with(
            endpoint=self.base_url,
            credential=self.mock_credential.return_value
        )

    def test_prepare_completion_params(self):
        """Test preparation of completion parameters."""
        # Create test messages
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello, how are you?")
        ]
        
        # Set up test parameters
        temperature = 0.7
        max_tokens = 100
        top_p = 1.0
        stop_sequences = ["STOP"]
        
        # Get the completion parameters
        params = self.provider.prepare_completion_params(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop_sequences=stop_sequences
        )
        
        # Assert that the parameters are correct
        self.assertEqual(params["deployment_id"], self.model)
        self.assertEqual(params["temperature"], temperature)
        self.assertEqual(params["max_tokens"], max_tokens)
        self.assertEqual(params["top_p"], top_p)
        self.assertEqual(params["stop"], stop_sequences)
        
        # Assert that the messages were converted correctly
        self.assertEqual(len(params["messages"]), 2)
        self.assertEqual(params["messages"][0]["role"], "system")
        self.assertEqual(params["messages"][0]["content"], "You are a helpful assistant.")
        self.assertEqual(params["messages"][1]["role"], "user")
        self.assertEqual(params["messages"][1]["content"], "Hello, how are you?")

    @patch('yamllm.providers.azure_foundry_provider.Console')
    def test_handle_non_streaming_response(self, mock_console):
        """Test handling of non-streaming responses."""
        # Create test messages
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello, how are you?")
        ]
        
        # Set up test parameters
        params = {
            "deployment_id": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 1.0
        }
        
        # Set up mock response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "I'm doing well, thank you!"
        self.provider.client.chat_completions.create.return_value = mock_response
        
        # Call the method
        response = self.provider.handle_non_streaming_response(messages, params)
        
        # Assert that the response is correct
        self.assertEqual(response, "I'm doing well, thank you!")
        
        # Assert that the client was called with the correct parameters
        expected_params = params.copy()
        deployment_id = expected_params.pop("deployment_id")
        self.provider.client.chat_completions.create.assert_called_with(
            deployment_name=deployment_id,
            **expected_params
        )

    @patch('yamllm.providers.azure_foundry_provider.Console')
    def test_handle_non_streaming_response_with_tools(self, mock_console):
        """Test handling of non-streaming responses with tools."""
        # Create test messages
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="What's the weather in New York?")
        ]
        
        # Create test tools
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The location to get weather for"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
        
        # Set up test parameters
        params = {
            "deployment_id": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 1.0
        }
        
        # Create a tool call in the mock response
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = '{"location": "New York"}'
        
        # Set up mock response with tool calls
        mock_response = MagicMock()
        mock_response.choices[0].message.tool_calls = [mock_tool_call]
        mock_response.choices[0].message.content = None
        self.provider.client.chat_completions.create.return_value = mock_response
        
        # Call the method
        response = self.provider.handle_non_streaming_response(messages, params, tools)
        
        # Assert that the response is the model message with tool calls
        self.assertEqual(response, mock_response.choices[0].message)
        
        # Assert that the client was called with the correct parameters
        expected_params = params.copy()
        deployment_id = expected_params.pop("deployment_id")
        expected_params["tools"] = tools
        expected_params["tool_choice"] = "auto"
        self.provider.client.chat_completions.create.assert_called_with(
            deployment_name=deployment_id,
            **expected_params
        )

    def test_create_embedding(self):
        """Test creation of embeddings."""
        # Set up test text
        text = "This is a test"
        
        # Set up mock response
        mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_response = MagicMock()
        mock_response.data[0].embedding = mock_embedding
        self.provider.client.embeddings.create.return_value = mock_response
        
        # Call the method
        embedding = self.provider.create_embedding(text)
        
        # Assert that the client was called with the correct parameters
        self.provider.client.embeddings.create.assert_called_with(
            deployment_name="text-embedding-ada-002",
            input=text
        )
        
        # Check that the returned embedding is correct (converting back from bytes)
        import numpy as np
        result_embedding = np.frombuffer(embedding, dtype=np.float32)
        self.assertEqual(list(result_embedding), mock_embedding)


if __name__ == '__main__':
    unittest.main()