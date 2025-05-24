"""
OpenAI provider implementation for YAMLLM.

This module implements the BaseProvider interface for OpenAI's API,
supporting the latest features including the response API.
"""

import json
from typing import Dict, List, Any, Optional, Union, Tuple, Iterator

from openai import OpenAI, OpenAIError
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from yamllm.core.providers.base import BaseProvider


class OpenAIProvider(BaseProvider):
    """
    OpenAI provider implementation using the latest OpenAI API.
    
    This class implements the BaseProvider interface for OpenAI models,
    supporting chat completions, embeddings, and tool calling.
    """
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        """
        Initialize the OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            base_url: Optional base URL for the API endpoint
            **kwargs: Additional OpenAI-specific parameters
        """
        self.api_key = api_key
        self.base_url = base_url
        
        # Initialize the OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # Initialize a separate client for embeddings
        self.embedding_client = OpenAI(
            api_key=self.api_key
        )
    
    def get_completion(self, 
                      messages: List[Dict[str, Any]], 
                      model: str,
                      temperature: float,
                      max_tokens: int,
                      top_p: float,
                      stop_sequences: Optional[List[str]] = None,
                      tools: Optional[List[Dict[str, Any]]] = None,
                      stream: bool = False,
                      **kwargs) -> ChatCompletion:
        """
        Get a completion from OpenAI.
        
        Args:
            messages: List of message objects with role and content
            model: OpenAI model identifier
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter
            stop_sequences: Optional list of stop sequences
            tools: Optional list of tool definitions
            stream: Whether to stream the response
            **kwargs: Additional OpenAI-specific parameters
            
        Returns:
            OpenAI ChatCompletion object
        """
        # Prepare the request parameters
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": stream
        }
        
        # Add optional parameters if provided
        if stop_sequences:
            params["stop"] = stop_sequences
        
        # Add tools if provided
        if tools:
            params["tools"] = tools
            params["tool_choice"] = kwargs.get("tool_choice", "auto")
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in params and key != "tool_choice":
                params[key] = value
        
        # Make the API request
        return self.client.chat.completions.create(**params)
    
    def get_streaming_completion(self, 
                               messages: List[Dict[str, Any]], 
                               model: str,
                               temperature: float,
                               max_tokens: int,
                               top_p: float,
                               stop_sequences: Optional[List[str]] = None,
                               tools: Optional[List[Dict[str, Any]]] = None,
                               **kwargs) -> Iterator[ChatCompletionChunk]:
        """
        Get a streaming completion from OpenAI.
        
        Args:
            messages: List of message objects with role and content
            model: OpenAI model identifier
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter
            stop_sequences: Optional list of stop sequences
            tools: Optional list of tool definitions
            **kwargs: Additional OpenAI-specific parameters
            
        Returns:
            Iterator of OpenAI ChatCompletionChunk objects
        """
        # Set stream=True and call get_completion to get a streaming response
        return self.get_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop_sequences=stop_sequences,
            tools=tools,
            stream=True,
            **kwargs
        )
    
    def create_embedding(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """
        Create an embedding for the given text using OpenAI.
        
        Args:
            text: Text to embed
            model: Model to use for embedding (default: text-embedding-3-small)
            
        Returns:
            Embedding vector as a list of floats
        """
        response = self.embedding_client.embeddings.create(
            input=text,
            model=model
        )
        
        return response.data[0].embedding
    
    def format_tool_calls(self, tool_calls: Any) -> List[Dict[str, Any]]:
        """
        Format OpenAI tool calls to standardized format.
        
        Args:
            tool_calls: OpenAI tool calls object
            
        Returns:
            List of standardized tool call objects
        """
        if not tool_calls:
            return []
        
        formatted_calls = []
        for tool_call in tool_calls:
            formatted_call = {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }
            }
            formatted_calls.append(formatted_call)
        
        return formatted_calls
    
    def format_tool_results(self, tool_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format tool results for OpenAI.
        
        Args:
            tool_results: List of standardized tool result objects
            
        Returns:
            List of OpenAI-compatible tool result objects
        """
        formatted_results = []
        for result in tool_results:
            formatted_result = {
                "role": "tool",
                "tool_call_id": result.get("tool_call_id"),
                "content": result.get("content")
            }
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    def close(self):
        """
        Close the OpenAI client and release resources.
        """
        if hasattr(self, 'client'):
            self.client.close()
        
        if hasattr(self, 'embedding_client'):
            self.embedding_client.close()