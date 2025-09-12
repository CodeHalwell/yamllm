"""
Async Anthropic provider implementation for YAMLLM.

This module implements the AsyncBaseProvider interface for Anthropic's Claude API
with full async/await support.
"""

import json
import logging
from typing import Dict, List, Any, Optional, AsyncIterator

from anthropic import AsyncAnthropic
from anthropic.types import Message

from yamllm.providers.async_base import AsyncBaseProvider
from yamllm.providers.exceptions import ProviderError


logger = logging.getLogger(__name__)


class AsyncAnthropicProvider(AsyncBaseProvider):
    """
    Async Anthropic provider implementation.
    
    This class provides full async support for Anthropic Claude API calls.
    """
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        """
        Initialize the async Anthropic provider.
        
        Args:
            api_key: Anthropic API key
            base_url: Optional base URL for the API
            **kwargs: Additional Anthropic-specific parameters
        """
        self.api_key = api_key
        self.base_url = base_url
        self.client = None
        self.max_retries = kwargs.get('max_retries', 2)
    
    async def __aenter__(self):
        """Initialize async client."""
        self.client = AsyncAnthropic(
            api_key=self.api_key,
            base_url=self.base_url,
            max_retries=self.max_retries
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up async client."""
        if self.client:
            await self.client.close()
    
    async def get_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        **kwargs
    ) -> Message:
        """
        Get an async completion from Anthropic.
        
        Args:
            messages: List of message objects
            model: Anthropic model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stop_sequences: Optional stop sequences
            tools: Optional tool definitions
            tool_choice: Tool choice strategy
            **kwargs: Additional parameters
            
        Returns:
            Anthropic Message object
        """
        if not self.client:
            raise ProviderError("Anthropic", "Client not initialized. Use async context manager.")
        
        # Convert messages to Anthropic format
        anthropic_messages = self._convert_messages(messages)
        
        # Extract system message if present
        system_message = None
        if anthropic_messages and anthropic_messages[0]["role"] == "system":
            system_message = anthropic_messages.pop(0)["content"]
        
        # Prepare parameters
        params = {
            "model": model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p
        }
        
        if system_message:
            params["system"] = system_message
        
        if stop_sequences:
            params["stop_sequences"] = stop_sequences
        
        # Add tools if provided
        if tools:
            params["tools"] = self._convert_tools(tools)
            if tool_choice != "auto":
                params["tool_choice"] = {"type": tool_choice}
        
        # Add any additional parameters
        params.update(kwargs)
        
        try:
            response = await self.client.messages.create(**params)
            
            # Convert to dict format with tool calls if present
            result = {
                "content": response.content[0].text if response.content else "",
                "usage": {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                }
            }
            
            # Check for tool use
            if response.content and any(block.type == "tool_use" for block in response.content):
                tool_calls = []
                for block in response.content:
                    if block.type == "tool_use":
                        tool_calls.append({
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.input)
                            }
                        })
                result["tool_calls"] = tool_calls
            
            return result
            
        except Exception as e:
            logger.error(f"Anthropic async API error: {str(e)}")
            raise ProviderError("Anthropic", f"API error: {str(e)}") from e
    
    async def get_streaming_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Get an async streaming completion from Anthropic.
        
        Args:
            messages: List of message objects
            model: Anthropic model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stop_sequences: Optional stop sequences
            tools: Optional tool definitions
            tool_choice: Tool choice strategy
            **kwargs: Additional parameters
            
        Yields:
            Text chunks from the response
        """
        if not self.client:
            raise ProviderError("Anthropic", "Client not initialized. Use async context manager.")
        
        # Convert messages
        anthropic_messages = self._convert_messages(messages)
        
        # Extract system message
        system_message = None
        if anthropic_messages and anthropic_messages[0]["role"] == "system":
            system_message = anthropic_messages.pop(0)["content"]
        
        # Prepare parameters
        params = {
            "model": model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": True
        }
        
        if system_message:
            params["system"] = system_message
        
        if stop_sequences:
            params["stop_sequences"] = stop_sequences
        
        if tools:
            params["tools"] = self._convert_tools(tools)
            if tool_choice != "auto":
                params["tool_choice"] = {"type": tool_choice}
        
        params.update(kwargs)
        
        try:
            async with self.client.messages.stream(**params) as stream:
                async for event in stream:
                    if event.type == "content_block_delta":
                        if hasattr(event.delta, 'text'):
                            yield event.delta.text
                            
        except Exception as e:
            logger.error(f"Anthropic async streaming error: {str(e)}")
            raise ProviderError("Anthropic", f"Streaming error: {str(e)}") from e
    
    def _convert_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Convert messages to Anthropic format."""
        anthropic_messages = []
        
        for msg in messages:
            role = msg["role"]
            
            # Convert 'system' to first message format (handled separately)
            if role == "system":
                anthropic_messages.append(msg)
            # Convert 'tool' to 'user' with tool result
            elif role == "tool":
                anthropic_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.get("tool_call_id"),
                            "content": msg["content"]
                        }
                    ]
                })
            else:
                # Handle assistant messages with tool calls
                if role == "assistant" and "tool_calls" in msg:
                    content = []
                    if msg.get("content"):
                        content.append({"type": "text", "text": msg["content"]})
                    
                    for tool_call in msg["tool_calls"]:
                        content.append({
                            "type": "tool_use",
                            "id": tool_call["id"],
                            "name": tool_call["function"]["name"],
                            "input": json.loads(tool_call["function"]["arguments"])
                        })
                    
                    anthropic_messages.append({
                        "role": role,
                        "content": content
                    })
                else:
                    anthropic_messages.append(msg)
        
        return anthropic_messages
    
    def _convert_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert tools to Anthropic format."""
        anthropic_tools = []
        
        for tool in tools:
            if tool["type"] == "function":
                anthropic_tools.append({
                    "name": tool["function"]["name"],
                    "description": tool["function"]["description"],
                    "input_schema": tool["function"]["parameters"]
                })
        
        return anthropic_tools
    
    def format_tool_calls(self, tool_calls: Any) -> List[Dict[str, Any]]:
        """Format Anthropic tool calls to standardized format."""
        # Already formatted in get_completion
        return tool_calls