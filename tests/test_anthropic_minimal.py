"""
Minimal test for the Anthropic provider implementation.
"""

import unittest
from unittest.mock import patch
import sys
import os

# Add the project directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the AnthropicProvider
from yamllm.providers.anthropic import AnthropicProvider


class TestAnthropicProvider(unittest.TestCase):
    
    @patch('yamllm.providers.anthropic.Anthropic')
    def test_init(self, mock_anthropic):
        """Test initializing the AnthropicProvider."""
        provider = AnthropicProvider(api_key="test_key", base_url="https://test.api")
        
        # Check that the Anthropic client was initialized correctly
        mock_anthropic.assert_called_once_with(
            api_key="test_key",
            base_url="https://test.api"
        )
    
    @patch('yamllm.providers.anthropic.Anthropic')
    def test_message_conversion(self, _):
        """Test converting messages to Anthropic format."""
        provider = AnthropicProvider(api_key="test_key")
        
        # Test basic message conversion
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        
        anthropic_messages = provider._convert_messages_to_anthropic_format(messages)
        
        # Check that the system message is converted properly
        self.assertEqual(anthropic_messages[0]["role"], "user")
        self.assertIn("<s>You are a helpful assistant</s>", anthropic_messages[0]["content"])
        
        # Check user message
        self.assertEqual(anthropic_messages[1]["role"], "user")
        self.assertEqual(anthropic_messages[1]["content"], "Hello")
        
        # Check assistant message
        self.assertEqual(anthropic_messages[2]["role"], "assistant")
        self.assertEqual(anthropic_messages[2]["content"], "Hi there")


if __name__ == '__main__':
    unittest.main()