"""
Test for the Anthropic provider implementation.
"""

import pytest
import os
import sys
from unittest.mock import MagicMock, patch

# Mock the dependencies we don't need for our unit tests
sys.modules['faiss'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['sqlite3'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['sklearn'] = MagicMock()
sys.modules['scikit-learn'] = MagicMock()
sys.modules['pickle'] = MagicMock()
sys.modules['rich'] = MagicMock()
sys.modules['openai'] = MagicMock()

# Directly import AnthropicProvider without going through the main module
sys.path.append('/home/runner/work/yamllm/yamllm')
from yamllm.core.providers.base import BaseProvider
from yamllm.core.providers.anthropic import AnthropicProvider


def test_anthropic_provider_init():
    """Test initializing the AnthropicProvider."""
    with patch('anthropic.Anthropic') as mock_anthropic:
        provider = AnthropicProvider(api_key="test_key", base_url="https://test.api")
        
        # Check that the Anthropic client was initialized correctly
        mock_anthropic.assert_called_once_with(
            api_key="test_key",
            base_url="https://test.api"
        )


def test_anthropic_provider_message_conversion():
    """Test converting messages to Anthropic format."""
    with patch('anthropic.Anthropic'):
        provider = AnthropicProvider(api_key="test_key")
        
        # Test basic message conversion
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        
        anthropic_messages = provider._convert_messages_to_anthropic_format(messages)
        
        # Check that the system message is converted properly
        assert anthropic_messages[0]["role"] == "user"
        assert "<s>You are a helpful assistant</s>" in anthropic_messages[0]["content"]
        
        # Check user message
        assert anthropic_messages[1]["role"] == "user"
        assert anthropic_messages[1]["content"] == "Hello"
        
        # Check assistant message
        assert anthropic_messages[2]["role"] == "assistant"
        assert anthropic_messages[2]["content"] == "Hi there"


def test_anthropic_provider_tool_conversion():
    """Test converting tools to Anthropic format."""
    with patch('anthropic.Anthropic'):
        provider = AnthropicProvider(api_key="test_key")
        
        # Test tool conversion
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
        
        anthropic_tools = provider._convert_tools_to_anthropic_format(tools)
        
        # Check the converted tool
        assert len(anthropic_tools) == 1
        assert anthropic_tools[0]["name"] == "get_weather"
        assert anthropic_tools[0]["description"] == "Get the weather for a location"
        assert "properties" in anthropic_tools[0]["input_schema"]
        assert "location" in anthropic_tools[0]["input_schema"]["properties"]