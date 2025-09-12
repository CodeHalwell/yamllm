import pytest
from unittest.mock import MagicMock, patch
import json

from yamllm.providers.openai import OpenAIProvider


class TestOpenAIToolFormatting:
    
    @pytest.fixture
    def mock_openai_provider(self):
        """Create a mock OpenAI provider"""
        with patch('openai.OpenAI'):
            provider = OpenAIProvider(api_key="fake-key", model="fake-model")
            return provider
    
    def test_format_tool_calls(self, mock_openai_provider):
        """Test formatting tool calls in OpenAI provider"""
        # Create a mock tool call object
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "web_search"
        mock_tool_call.function.arguments = json.dumps({"query": "current weather in New York"})
        
        # Test formatting
        formatted_calls = mock_openai_provider.format_tool_calls([mock_tool_call])
        
        # Verify the formatting is correct
        assert len(formatted_calls) == 1
        assert formatted_calls[0]["id"] == "call_123"
        assert formatted_calls[0]["type"] == "function"
        assert formatted_calls[0]["function"]["name"] == "web_search"
        assert json.loads(formatted_calls[0]["function"]["arguments"]) == {"query": "current weather in New York"}
    
    def test_format_tool_results(self, mock_openai_provider):
        """Test formatting tool results in OpenAI provider"""
        # Sample tool result
        tool_result = [
            {
                "tool_call_id": "call_123",
                "name": "web_search",
                "content": "The current weather in New York is 75°F and sunny."
            }
        ]
        
        # Test formatting
        formatted_results = mock_openai_provider.format_tool_results(tool_result)
        
        # Verify the formatting is correct
        assert len(formatted_results) == 1
        assert formatted_results[0]["role"] == "tool"
        assert formatted_results[0]["tool_call_id"] == "call_123"
        assert formatted_results[0]["content"] == "The current weather in New York is 75°F and sunny."