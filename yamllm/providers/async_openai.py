"""
Async OpenAI provider implementation for YAMLLM.

This module implements the AsyncBaseProvider interface for OpenAI's API
with full async/await support.
"""

import logging
from typing import Dict, List, Any, Optional, AsyncIterator

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from yamllm.providers.async_base import AsyncBaseProvider
from yamllm.providers.exceptions import ProviderError


logger = logging.getLogger(__name__)


class AsyncOpenAIProvider(AsyncBaseProvider):
    """
    Async OpenAI provider implementation.
    
    This class provides full async support for OpenAI API calls,
    enabling better performance when handling multiple requests.
    """
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        """
        Initialize the async OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            base_url: Optional base URL for the API
            **kwargs: Additional OpenAI-specific parameters
        """
        self.api_key = api_key
        self.base_url = base_url
        self.client = None
        self.embedding_client = None
    
    async def __aenter__(self):
        """Initialize async clients."""
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        self.embedding_client = AsyncOpenAI(api_key=self.api_key)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up async clients."""
        if self.client:
            await self.client.close()
        if self.embedding_client:
            await self.embedding_client.close()
    
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
    ) -> ChatCompletion:
        """
        Get an async completion from OpenAI.
        
        Args:
            messages: List of message objects
            model: OpenAI model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stop_sequences: Optional stop sequences
            tools: Optional tool definitions
            tool_choice: Tool choice strategy
            **kwargs: Additional parameters
            
        Returns:
            OpenAI ChatCompletion object
        """
        if not self.client:
            raise ProviderError("OpenAI", "Client not initialized. Use async context manager.")
        
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": False
        }
        
        if stop_sequences:
            params["stop"] = stop_sequences
        
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice
        
        # Add any additional parameters
        params.update(kwargs)
        
        try:
            return await self.client.chat.completions.create(**params)
        except Exception as e:
            logger.error(f"OpenAI async API error: {str(e)}")
            raise ProviderError("OpenAI", f"API error: {str(e)}") from e
    
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
    ) -> AsyncIterator[ChatCompletionChunk]:
        """
        Get an async streaming completion from OpenAI.
        
        Args:
            messages: List of message objects
            model: OpenAI model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stop_sequences: Optional stop sequences
            tools: Optional tool definitions
            tool_choice: Tool choice strategy
            **kwargs: Additional parameters
            
        Yields:
            ChatCompletionChunk objects
        """
        if not self.client:
            raise ProviderError("OpenAI", "Client not initialized. Use async context manager.")
        
        params = {
            "model": model,
            "messages": messages,
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
        
        # Add any additional parameters
        params.update(kwargs)
        
        try:
            async for chunk in await self.client.chat.completions.create(**params):
                yield chunk
        except Exception as e:
            logger.error(f"OpenAI async streaming error: {str(e)}")
            raise ProviderError("OpenAI", f"Streaming error: {str(e)}") from e
    
    async def create_embedding(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """
        Create an async embedding.
        
        Args:
            text: Text to embed
            model: Embedding model to use
            
        Returns:
            Embedding vector
        """
        if not self.embedding_client:
            raise ProviderError("OpenAI", "Embedding client not initialized")
        
        try:
            response = await self.embedding_client.embeddings.create(
                input=text,
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI async embedding error: {str(e)}")
            raise ProviderError("OpenAI", f"Embedding error: {str(e)}") from e
    
    def format_tool_calls(self, tool_calls: Any) -> List[Dict[str, Any]]:
        """Format OpenAI tool calls to standardized format."""
        if not tool_calls:
            return []
        
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