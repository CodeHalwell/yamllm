import pytest
from unittest.mock import MagicMock, patch
import json

from yamllm.providers.base import ToolDefinition, ToolCall
from yamllm.providers.openai_provider import OpenAIProvider
from yamllm.providers.google_provider import GoogleGeminiProvider
from yamllm.providers.mistral_provider import MistralProvider
from yamllm.providers.deepseek_provider import DeepSeekProvider
from yamllm.providers.azure_openai_provider import AzureOpenAIProvider
from yamllm.providers.azure_foundry_provider import AzureFoundryProvider


class TestToolFormatting:
    
    @pytest.fixture
    def tool_calls_data(self):
        """Sample tool calls data for testing"""
        return [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "web_search",
                    "arguments": json.dumps({"query": "current weather in New York"})
                }
            }
        ]
    
    @pytest.fixture
    def tool_results_data(self):
        """Sample tool results data for testing"""
        return [
            {
                "tool_call_id": "call_123",
                "name": "web_search",
                "content": "The current weather in New York is 75°F and sunny."
            }
        ]
    
    def test_openai_format_tool_calls(self, tool_calls_data):
        """Test OpenAI provider tool call formatting"""
        # Setup
        provider = OpenAIProvider(api_key="fake-key", model="fake-model")
        
        # Create a mock OpenAI tool call object
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "web_search"
        mock_tool_call.function.arguments = json.dumps({"query": "current weather in New York"})
        
        # Test
        formatted_calls = provider.format_tool_calls([mock_tool_call])
        
        # Verify
        assert len(formatted_calls) == 1
        assert formatted_calls[0]["id"] == "call_123"
        assert formatted_calls[0]["type"] == "function"
        assert formatted_calls[0]["function"]["name"] == "web_search"
        assert json.loads(formatted_calls[0]["function"]["arguments"]) == {"query": "current weather in New York"}
    
    def test_openai_format_tool_results(self, tool_results_data):
        """Test OpenAI provider tool results formatting"""
        # Setup
        provider = OpenAIProvider(api_key="fake-key", model="fake-model")
        
        # Test
        formatted_results = provider.format_tool_results(tool_results_data)
        
        # Verify
        assert len(formatted_results) == 1
        assert formatted_results[0]["role"] == "tool"
        assert formatted_results[0]["tool_call_id"] == "call_123"
        assert formatted_results[0]["content"] == "The current weather in New York is 75°F and sunny."
    
    def test_google_format_tool_calls(self):
        """Test Google Gemini provider tool call formatting"""
        # Setup
        provider = GoogleGeminiProvider(api_key="fake-key", model="fake-model")
        
        # Create a mock Google function call object
        mock_function_call = MagicMock()
        mock_function_call.function_call.name = "web_search"
        mock_function_call.function_call.args = json.dumps({"query": "current weather in New York"})
        
        # Test
        formatted_calls = provider.format_tool_calls([mock_function_call])
        
        # Verify
        assert len(formatted_calls) == 1
        assert formatted_calls[0]["id"].startswith("call_")
        assert formatted_calls[0]["type"] == "function"
        assert formatted_calls[0]["function"]["name"] == "web_search"
        assert json.loads(formatted_calls[0]["function"]["arguments"]) == {"query": "current weather in New York"}
    
    def test_google_format_tool_results(self, tool_results_data):
        """Test Google Gemini provider tool results formatting"""
        # Setup
        provider = GoogleGeminiProvider(api_key="fake-key", model="fake-model")
        
        # Test
        formatted_results = provider.format_tool_results(tool_results_data)
        
        # Verify
        assert len(formatted_results) == 1
        assert formatted_results[0]["role"] == "user"
        assert "function_response" in formatted_results[0]["parts"][0]
        assert formatted_results[0]["parts"][0]["function_response"]["name"] == "web_search"
        assert formatted_results[0]["parts"][0]["function_response"]["response"]["content"] == "The current weather in New York is 75°F and sunny."
    
    def test_mistral_format_tool_calls(self, tool_calls_data):
        """Test Mistral provider tool call formatting"""
        # Setup
        provider = MistralProvider(api_key="fake-key", model="fake-model")
        
        # Create a mock Mistral tool call object (similar to OpenAI format)
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "web_search"
        mock_tool_call.function.arguments = json.dumps({"query": "current weather in New York"})
        
        # Test
        formatted_calls = provider.format_tool_calls([mock_tool_call])
        
        # Verify
        assert len(formatted_calls) == 1
        assert formatted_calls[0]["id"] == "call_123"
        assert formatted_calls[0]["type"] == "function"
        assert formatted_calls[0]["function"]["name"] == "web_search"
        assert json.loads(formatted_calls[0]["function"]["arguments"]) == {"query": "current weather in New York"}
    
    def test_mistral_format_tool_results(self, tool_results_data):
        """Test Mistral provider tool results formatting"""
        # Setup
        provider = MistralProvider(api_key="fake-key", model="fake-model")
        
        # Test
        formatted_results = provider.format_tool_results(tool_results_data)
        
        # Verify
        assert len(formatted_results) == 1
        assert formatted_results[0]["role"] == "tool"
        assert formatted_results[0]["tool_call_id"] == "call_123"
        assert formatted_results[0]["name"] == "web_search"
        assert formatted_results[0]["content"] == "The current weather in New York is 75°F and sunny."
    
    def test_deepseek_provider_inheritance(self):
        """Test DeepSeek provider inherits OpenAI provider tool formatting methods"""
        # Setup
        openai_provider = OpenAIProvider(api_key="fake-key", model="fake-model")
        deepseek_provider = DeepSeekProvider(api_key="fake-key", model="fake-model")
        
        # Create a mock tool call object
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "web_search"
        mock_tool_call.function.arguments = json.dumps({"query": "current weather in New York"})
        
        # Test format_tool_calls
        openai_result = openai_provider.format_tool_calls([mock_tool_call])
        deepseek_result = deepseek_provider.format_tool_calls([mock_tool_call])
        
        # Verify both providers format tool calls in the same way
        assert openai_result[0]["id"] == deepseek_result[0]["id"]
        assert openai_result[0]["type"] == deepseek_result[0]["type"]
        assert openai_result[0]["function"]["name"] == deepseek_result[0]["function"]["name"]
        assert openai_result[0]["function"]["arguments"] == deepseek_result[0]["function"]["arguments"]
        
        # Test format_tool_results
        tool_result = [{"tool_call_id": "call_123", "content": "Result content"}]
        openai_result = openai_provider.format_tool_results(tool_result)
        deepseek_result = deepseek_provider.format_tool_results(tool_result)
        
        # Verify both providers format tool results in the same way
        assert openai_result[0]["role"] == deepseek_result[0]["role"]
        assert openai_result[0]["tool_call_id"] == deepseek_result[0]["tool_call_id"]
        assert openai_result[0]["content"] == deepseek_result[0]["content"]
    
    def test_azure_openai_format_tool_calls(self, tool_calls_data):
        """Test Azure OpenAI provider tool call formatting"""
        # Setup
        provider = AzureOpenAIProvider(api_key="fake-key", model="fake-model", base_url="https://example.com")
        
        # Create a mock Azure OpenAI tool call object
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "web_search"
        mock_tool_call.function.arguments = json.dumps({"query": "current weather in New York"})
        
        # Test
        formatted_calls = provider.format_tool_calls([mock_tool_call])
        
        # Verify
        assert len(formatted_calls) == 1
        assert formatted_calls[0]["id"] == "call_123"
        assert formatted_calls[0]["type"] == "function"
        assert formatted_calls[0]["function"]["name"] == "web_search"
        assert json.loads(formatted_calls[0]["function"]["arguments"]) == {"query": "current weather in New York"}
    
    def test_azure_foundry_format_tool_calls(self, tool_calls_data):
        """Test Azure Foundry provider tool call formatting"""
        # Setup
        with patch("azure.ai.inference.InferenceClient"):
            with patch("azure.identity.DefaultAzureCredential"):
                provider = AzureFoundryProvider(api_key="fake-key", model="fake-model", base_url="https://example.com")
                
                # Create a mock Azure Foundry tool call object
                mock_tool_call = MagicMock()
                mock_tool_call.id = "call_123"
                mock_tool_call.function.name = "web_search"
                mock_tool_call.function.arguments = json.dumps({"query": "current weather in New York"})
                
                # Test
                formatted_calls = provider.format_tool_calls([mock_tool_call])
                
                # Verify
                assert len(formatted_calls) == 1
                assert formatted_calls[0]["id"] == "call_123"
                assert formatted_calls[0]["type"] == "function"
                assert formatted_calls[0]["function"]["name"] == "web_search"
                assert json.loads(formatted_calls[0]["function"]["arguments"]) == {"query": "current weather in New York"}