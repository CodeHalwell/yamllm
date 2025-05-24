"""
Base provider interface for YAMLLM.

This module defines the abstract base class that all provider
implementations must inherit from.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple


class BaseProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    This class defines the interface that all provider implementations
    must follow to ensure consistent behavior across different LLM services.
    """
    
    @abstractmethod
    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        """
        Initialize the provider with authentication credentials.
        
        Args:
            api_key: API key for the provider
            base_url: Optional base URL for the API endpoint
            **kwargs: Additional provider-specific parameters
        """
        pass
    
    @abstractmethod
    def get_completion(self, 
                      messages: List[Dict[str, Any]], 
                      model: str,
                      temperature: float,
                      max_tokens: int,
                      top_p: float,
                      stop_sequences: Optional[List[str]] = None,
                      tools: Optional[List[Dict[str, Any]]] = None,
                      stream: bool = False,
                      **kwargs) -> Dict[str, Any]:
        """
        Get a completion from the LLM.
        
        Args:
            messages: List of message objects with role and content
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter
            stop_sequences: Optional list of stop sequences
            tools: Optional list of tool definitions
            stream: Whether to stream the response
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Response from the LLM
        """
        pass
    
    @abstractmethod
    def get_streaming_completion(self, 
                               messages: List[Dict[str, Any]], 
                               model: str,
                               temperature: float,
                               max_tokens: int,
                               top_p: float,
                               stop_sequences: Optional[List[str]] = None,
                               tools: Optional[List[Dict[str, Any]]] = None,
                               **kwargs) -> Any:
        """
        Get a streaming completion from the LLM.
        
        Args:
            messages: List of message objects with role and content
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter
            stop_sequences: Optional list of stop sequences
            tools: Optional list of tool definitions
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Streaming response iterator
        """
        pass
    
    @abstractmethod
    def create_embedding(self, text: str, model: str) -> List[float]:
        """
        Create an embedding for the given text.
        
        Args:
            text: Text to embed
            model: Model to use for embedding
            
        Returns:
            Embedding vector
        """
        pass
    
    @abstractmethod
    def format_tool_calls(self, tool_calls: Any) -> List[Dict[str, Any]]:
        """
        Format tool calls from provider-specific format to standardized format.
        
        Args:
            tool_calls: Provider-specific tool calls object
            
        Returns:
            List of standardized tool call objects
        """
        pass
    
    @abstractmethod
    def format_tool_results(self, tool_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format tool results to provider-specific format.
        
        Args:
            tool_results: List of standardized tool result objects
            
        Returns:
            List of provider-specific tool result objects
        """
        pass
    
    @abstractmethod
    def close(self):
        """
        Close the provider client and release resources.
        """
        pass