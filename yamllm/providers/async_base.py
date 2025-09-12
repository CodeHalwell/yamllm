"""
Async base provider interface for YAMLLM.

This module defines the async base interface that providers can implement
for better performance and concurrency.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, AsyncIterator


class AsyncBaseProvider(ABC):
    """
    Async base provider interface for YAMLLM.
    
    This abstract class defines the async interface that LLM providers
    can implement for better performance.
    """
    
    @abstractmethod
    async def __aenter__(self):
        """Async context manager entry."""
        pass
    
    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass
    
    @abstractmethod
    async def get_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto"
    ) -> Any:
        """
        Get an async completion from the model.
        
        Args:
            messages: List of message objects with role and content
            model: The model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stop_sequences: Optional stop sequences
            tools: Optional tool definitions
            tool_choice: Tool choice strategy
            
        Returns:
            The response from the model
        """
        pass
    
    @abstractmethod
    async def get_streaming_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto"
    ) -> AsyncIterator[Any]:
        """
        Get an async streaming completion from the model.
        
        Args:
            messages: List of message objects with role and content
            model: The model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stop_sequences: Optional stop sequences
            tools: Optional tool definitions
            tool_choice: Tool choice strategy
            
        Returns:
            Async iterator of response chunks
        """
        pass
    
    async def create_embedding(self, text: str, model: str) -> List[float]:
        """
        Create an embedding for the given text.
        
        Args:
            text: Text to embed
            model: Embedding model to use
            
        Returns:
            Embedding vector
        """
        raise NotImplementedError("This provider does not support embeddings")
    
    def format_tool_calls(self, tool_calls: Any) -> List[Dict[str, Any]]:
        """
        Format tool calls into standardized format.
        
        Args:
            tool_calls: Provider-specific tool calls
            
        Returns:
            Standardized tool call format
        """
        return tool_calls