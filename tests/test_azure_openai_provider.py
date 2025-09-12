"""
Tests for Azure OpenAI provider.

This module contains tests for the Azure OpenAI provider implementation.
"""

import unittest
from unittest.mock import MagicMock, patch

from yamllm.providers.azure_openai import AzureOpenAIProvider


class TestAzureOpenAIProvider(unittest.TestCase):
    """Test cases for AzureOpenAIProvider."""

    def setUp(self):
        """Set up test fixtures."""
        self.api_key = "test_api_key"
        self.model = "gpt-4"
        self.base_url = "https://test-endpoint.openai.azure.com"
        self.api_version = "2023-05-15"
        
        # Create a mock for the AzureOpenAI client (core path)
        self.mock_client_patcher = patch('yamllm.providers.azure_openai.AzureOpenAI')
        self.mock_client = self.mock_client_patcher.start()
        
        # Set up the provider (core constructor doesn't take model)
        self.provider = AzureOpenAIProvider(
            api_key=self.api_key,
            base_url=self.base_url,
            api_version=self.api_version
        )

    def tearDown(self):
        """Tear down test fixtures."""
        self.mock_client_patcher.stop()

    def test_init(self):
        """Test initialization of the provider."""
        # Assert that the provider was initialized correctly
        self.assertEqual(self.provider.api_key, self.api_key)
        self.assertEqual(self.provider.base_url, self.base_url)
        self.assertEqual(self.provider.api_version, self.api_version)
        
        # Assert that the AzureOpenAI client was initialized with the correct parameters
        self.mock_client.assert_called_with(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.base_url
        )

    def test_get_completion_forwards_params(self):
        """Test that get_completion forwards parameters to SDK correctly."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ]
        temperature = 0.7
        max_tokens = 100
        top_p = 1.0
        stop_sequences = ["STOP"]

        # Call get_completion
        self.provider.get_completion(
            messages=messages,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop_sequences=stop_sequences,
        )
        # Verify call to Azure SDK
        self.provider.client.chat.completions.create.assert_called_with(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=False,
            stop=stop_sequences
        )

    def test_get_completion_with_tools(self):
        """Test get_completion includes tool params when provided."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather in New York?"},
        ]
        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]

        self.provider.get_completion(
            messages=messages,
            model=self.model,
            temperature=0.7,
            max_tokens=100,
            top_p=1.0,
            tools=tools,
            tool_choice="auto",
        )

        self.provider.client.chat.completions.create.assert_called_with(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=100,
            top_p=1.0,
            stream=False,
            tools=tools,
            tool_choice="auto",
        )

    

    def test_create_embedding(self):
        """Test creation of embeddings."""
        text = "This is a test"
        mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_response = MagicMock()
        mock_response.data[0].embedding = mock_embedding
        self.provider.embedding_client.embeddings.create.return_value = mock_response
        embedding = self.provider.create_embedding(text)
        self.provider.embedding_client.embeddings.create.assert_called_with(
            model="text-embedding-ada-002", input=text
        )
        self.assertEqual(embedding, mock_embedding)


if __name__ == '__main__':
    unittest.main()
