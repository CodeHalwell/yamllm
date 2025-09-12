import unittest
from unittest.mock import MagicMock, patch

from yamllm.providers.azure_foundry import AzureFoundryProvider


class TestAzureFoundryProvider(unittest.TestCase):
    """Test cases for AzureFoundryProvider."""

    def setUp(self):
        """Set up test fixtures."""
        self.api_key = "test_api_key"
        self.model = "my-gpt4-deployment"
        self.base_url = "https://test-project.azureai.azure.com"
        self.project_id = "test-project-id"
        
        # Create a mock for the ChatCompletionsClient
        self.mock_chat_client_patcher = patch('azure.ai.inference.ChatCompletionsClient')
        self.mock_chat_client = self.mock_chat_client_patcher.start()

        # Create a mock for the EmbeddingsClient
        self.mock_embeddings_client_patcher = patch('azure.ai.inference.EmbeddingsClient')
        self.mock_embeddings_client = self.mock_embeddings_client_patcher.start()
        
        # Create a mock for DefaultAzureCredential
        self.mock_credential_patcher = patch('azure.identity.DefaultAzureCredential')
        self.mock_credential = self.mock_credential_patcher.start()
        
        # Set up the provider
        self.provider = AzureFoundryProvider(
            api_key=self.api_key,
            base_url=self.base_url,
            project_id=self.project_id
        )

    def tearDown(self):
        """Tear down test fixtures."""
        self.mock_chat_client_patcher.stop()
        self.mock_embeddings_client_patcher.stop()
        self.mock_credential_patcher.stop()

    def test_init(self):
        """Test initialization of the provider."""
        # Assert that the provider was initialized correctly
        self.assertEqual(self.provider.api_key, self.api_key)
        self.assertEqual(self.provider.base_url, self.base_url)
        self.assertEqual(self.provider.project_id, self.project_id)
        
        # Assert that the clients were initialized with the correct parameters
        self.mock_chat_client.assert_called_with(endpoint=self.base_url, credential=unittest.mock.ANY)
        self.mock_embeddings_client.assert_called_with(endpoint=self.base_url, credential=unittest.mock.ANY)


    def test_init_with_default_credential(self):
        """Test initialization with DefaultAzureCredential."""
        # Create provider with 'default' api_key
        provider = AzureFoundryProvider(api_key="default", base_url=self.base_url)
        
        # Assert that DefaultAzureCredential was created
        self.mock_credential.assert_called_once()
        
        # Assert that clients were initialized with the credential
        self.mock_chat_client.assert_called_with(endpoint=self.base_url, credential=self.mock_credential.return_value)
        self.mock_embeddings_client.assert_called_with(endpoint=self.base_url, credential=self.mock_credential.return_value)

    def test_get_completion_forwards_params(self):
        """Test get_completion forwards parameters to Azure Inference SDK."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ]
        self.provider.get_completion(
            messages=messages,
            model=self.model,
            temperature=0.7,
            max_tokens=100,
            top_p=1.0,
            stop_sequences=["STOP"],
        )
        self.provider.chat_completions_client.create.assert_called_with(
            deployment_name=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=100,
            top_p=1.0,
            stop=["STOP"],
        )

    def test_non_streaming_like_response(self):
        """Test handling of non-streaming responses."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ]
        params = {
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 1.0,
        }
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "I'm doing well, thank you!"
        self.provider.chat_completions_client.create.return_value = mock_response
        out = self.provider.get_completion(model=self.model, **params)
        self.assertEqual(out.choices[0].message.content, "I'm doing well, thank you!")

    

    def test_create_embedding(self):
        """Test creation of embeddings."""
        text = "This is a test"
        mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_response = MagicMock()
        mock_response.data[0].embedding = mock_embedding
        self.provider.embeddings_client.create.return_value = mock_response
        embedding = self.provider.create_embedding(text)
        self.provider.embeddings_client.create.assert_called_with(
            deployment_name="text-embedding-ada-002", input=text
        )
        self.assertEqual(embedding, mock_embedding)


if __name__ == '__main__':
    unittest.main()
