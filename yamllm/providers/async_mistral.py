"""
Async Mistral provider implementation for YAMLLM.

This module implements the AsyncBaseProvider interface for Mistral's API
with full async/await support.
"""

import logging
from typing import Dict, List, Any, Optional, AsyncIterator

from mistralai.async_client import MistralAsyncClient
from mistralai.models.chat_completion import ChatMessage, ToolCall

from yamllm.providers.async_base import AsyncBaseProvider
from yamllm.providers.exceptions import ProviderError


logger = logging.getLogger(__name__)


class AsyncMistralProvider(AsyncBaseProvider):
    """
    Async Mistral provider implementation.
    
    This class provides full async support for Mistral API calls.
    """
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        """
        Initialize the async Mistral provider.
        
        Args:
            api_key: Mistral API key
            base_url: Optional base URL for the API
            **kwargs: Additional parameters
        """
        self.api_key = api_key
        self.base_url = base_url
        self.client = None
    
    async def __aenter__(self):
        """Initialize async client."""
        self.client = MistralAsyncClient(
            api_key=self.api_key,
            endpoint=self.base_url if self.base_url else "https://api.mistral.ai"
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up async client."""
        # MistralAsyncClient doesn't have explicit close method
        pass
    
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
    ) -> Any:
        """
        Get an async completion from Mistral.
        
        Args:
            messages: List of message objects
            model: Mistral model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stop_sequences: Optional stop sequences
            tools: Optional tool definitions
            tool_choice: Tool choice strategy
            **kwargs: Additional parameters
            
        Returns:
            Response object
        """
        if not self.client:
            raise ProviderError("Mistral", "Client not initialized. Use async context manager.")
        
        # Convert messages to Mistral format
        mistral_messages = self._convert_messages(messages)
        
        # Prepare parameters
        params = {
            "model": model,
            "messages": mistral_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p
        }
        
        # Add optional parameters
        if stop_sequences:
            params["stop"] = stop_sequences
        
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in params:
                params[key] = value
        
        try:
            response = await self.client.chat(**params)
            
            # Format response
            result = {
                "content": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
            # Check for tool calls
            if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
                result["tool_calls"] = self.format_tool_calls(response.choices[0].message.tool_calls)
            
            return response  # Return full response for compatibility
            
        except Exception as e:
            logger.error(f"Mistral async API error: {str(e)}")
            raise ProviderError("Mistral", f"API error: {str(e)}") from e
    
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
    ) -> AsyncIterator[Any]:
        """
        Get an async streaming completion from Mistral.
        
        Args:
            messages: List of message objects
            model: Mistral model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stop_sequences: Optional stop sequences
            tools: Optional tool definitions
            tool_choice: Tool choice strategy
            **kwargs: Additional parameters
            
        Yields:
            Response chunks
        """
        if not self.client:
            raise ProviderError("Mistral", "Client not initialized. Use async context manager.")
        
        # Convert messages
        mistral_messages = self._convert_messages(messages)
        
        # Prepare parameters
        params = {
            "model": model,
            "messages": mistral_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": True
        }
        
        if stop_sequences:
            params["stop"] = stop_sequences
        
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice
        
        for key, value in kwargs.items():
            if key not in params:
                params[key] = value
        
        try:
            async for chunk in self.client.chat_stream(**params):
                yield chunk
                
        except Exception as e:
            logger.error(f"Mistral async streaming error: {str(e)}")
            raise ProviderError("Mistral", f"Streaming error: {str(e)}") from e
    
    def _convert_messages(self, messages: List[Dict[str, str]]) -> List[ChatMessage]:
        """Convert messages to Mistral format."""
        mistral_messages = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            # Handle tool messages
            if role == "tool":
                # Mistral expects function results as user messages
                mistral_messages.append(ChatMessage(
                    role="user",
                    content=f"Function {msg.get('name', 'unknown')} returned: {content}"
                ))
            else:
                # Handle assistant messages with tool calls
                if role == "assistant" and "tool_calls" in msg:
                    tool_calls = []
                    for tc in msg["tool_calls"]:
                        tool_calls.append(ToolCall(
                            id=tc["id"],
                            type="function",
                            function={
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"]
                            }
                        ))
                    
                    mistral_messages.append(ChatMessage(
                        role=role,
                        content=content,
                        tool_calls=tool_calls
                    ))
                else:
                    mistral_messages.append(ChatMessage(role=role, content=content))
        
        return mistral_messages
    
    def format_tool_calls(self, tool_calls: List[ToolCall]) -> List[Dict[str, Any]]:
        """Format Mistral tool calls to standardized format."""
        formatted = []
        
        for tool_call in tool_calls:
            formatted.append({
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }
            })
        
        return formatted
    
    async def create_embedding(self, text: str, model: str = "mistral-embed") -> List[float]:
        """Create an embedding asynchronously."""
        if not self.client:
            raise ProviderError("Mistral", "Client not initialized")
        
        try:
            response = await self.client.embeddings(
                model=model,
                input=[text]
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Mistral async embedding error: {str(e)}")
            raise ProviderError("Mistral", f"Embedding error: {str(e)}") from e