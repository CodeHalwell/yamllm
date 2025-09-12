"""
Async Azure OpenAI provider implementation for YAMLLM.

This module implements the AsyncBaseProvider interface for Azure OpenAI Service
with full async/await support.
"""

import os
import logging
from typing import Dict, List, Any, Optional, AsyncIterator

from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from yamllm.providers.async_base import AsyncBaseProvider
from yamllm.providers.exceptions import ProviderError


logger = logging.getLogger(__name__)


class AsyncAzureOpenAIProvider(AsyncBaseProvider):
    """
    Async Azure OpenAI provider implementation.
    
    This class provides full async support for Azure OpenAI Service.
    """
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        """
        Initialize the async Azure OpenAI provider.
        
        Args:
            api_key: Azure OpenAI API key
            base_url: Azure OpenAI endpoint
            **kwargs: Additional parameters including:
                - api_version: API version (default: 2024-02-15-preview)
                - deployment_name: Deployment name
        """
        self.api_key = api_key
        self.azure_endpoint = base_url or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = kwargs.get("api_version", "2024-02-15-preview")
        self.deployment_name = kwargs.get("deployment_name", "")
        self.client = None
        self.embedding_client = None
    
    async def __aenter__(self):
        """Initialize async clients."""
        self.client = AsyncAzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version
        )
        
        # Separate client for embeddings if needed
        self.embedding_client = AsyncAzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version
        )
        
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
        Get an async completion from Azure OpenAI.
        
        Args:
            messages: List of message objects
            model: Model/deployment name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stop_sequences: Optional stop sequences
            tools: Optional tool definitions
            tool_choice: Tool choice strategy
            **kwargs: Additional parameters
            
        Returns:
            ChatCompletion object
        """
        if not self.client:
            raise ProviderError("Azure OpenAI", "Client not initialized. Use async context manager.")
        
        # Use deployment name if provided, otherwise use model
        deployment = self.deployment_name or model
        
        # Prepare parameters
        params = {
            "model": deployment,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": False
        }
        
        # Add optional parameters
        if stop_sequences:
            params["stop"] = stop_sequences
        
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in params and key not in ["deployment_name", "api_version"]:
                params[key] = value
        
        try:
            return await self.client.chat.completions.create(**params)
        except Exception as e:
            logger.error(f"Azure OpenAI async API error: {str(e)}")
            raise ProviderError("Azure OpenAI", f"API error: {str(e)}") from e
    
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
        Get an async streaming completion from Azure OpenAI.
        
        Args:
            messages: List of message objects
            model: Model/deployment name
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
            raise ProviderError("Azure OpenAI", "Client not initialized. Use async context manager.")
        
        # Use deployment name if provided
        deployment = self.deployment_name or model
        
        params = {
            "model": deployment,
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
        
        for key, value in kwargs.items():
            if key not in params and key not in ["deployment_name", "api_version"]:
                params[key] = value
        
        try:
            async for chunk in await self.client.chat.completions.create(**params):
                yield chunk
        except Exception as e:
            logger.error(f"Azure OpenAI async streaming error: {str(e)}")
            raise ProviderError("Azure OpenAI", f"Streaming error: {str(e)}") from e
    
    async def create_embedding(self, text: str, model: str = "text-embedding-ada-002") -> List[float]:
        """
        Create an embedding asynchronously.
        
        Args:
            text: Text to embed
            model: Embedding model/deployment name
            
        Returns:
            Embedding vector
        """
        if not self.embedding_client:
            raise ProviderError("Azure OpenAI", "Embedding client not initialized")
        
        try:
            # Use deployment name if it matches an embedding model
            deployment = self.deployment_name if "embedding" in self.deployment_name else model
            
            response = await self.embedding_client.embeddings.create(
                input=text,
                model=deployment
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Azure OpenAI async embedding error: {str(e)}")
            raise ProviderError("Azure OpenAI", f"Embedding error: {str(e)}") from e
    
    def format_tool_calls(self, tool_calls: Any) -> List[Dict[str, Any]]:
        """Format Azure OpenAI tool calls to standardized format."""
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