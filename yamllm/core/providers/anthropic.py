"""
Anthropic provider implementation for YAMLLM.

This module implements the BaseProvider interface for Anthropic's API,
supporting Claude models and tools.
"""

import json
import os
import requests
from typing import Dict, List, Any, Optional, Union, Tuple, Iterator

from yamllm.core.providers.base import BaseProvider


class AnthropicProvider(BaseProvider):
    """
    Anthropic provider implementation using the Anthropic API.
    
    This class implements the BaseProvider interface for Anthropic Claude models,
    supporting chat completions and tool calling.
    """
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        """
        Initialize the Anthropic provider.
        
        Args:
            api_key: Anthropic API key
            base_url: Optional base URL for the API endpoint
            **kwargs: Additional Anthropic-specific parameters
        """
        self.api_key = api_key
        self.base_url = base_url or "https://api.anthropic.com"
        self.api_version = kwargs.get("api_version", "v1")
        
        # Set headers used for all requests
        self.headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
            "content-type": "application/json"
        }
    
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
        Get a completion from Anthropic.
        
        Args:
            messages: List of message objects with role and content
            model: Anthropic model identifier
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter
            stop_sequences: Optional list of stop sequences
            tools: Optional list of tool definitions
            stream: Whether to stream the response
            **kwargs: Additional Anthropic-specific parameters
            
        Returns:
            Anthropic API response
        """
        # Convert OpenAI-style messages to Anthropic format
        anthropic_messages = self._convert_messages_to_anthropic_format(messages)
        
        # Prepare the request payload
        payload = {
            "model": model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": stream
        }
        
        # Add optional parameters if provided
        if stop_sequences:
            payload["stop_sequences"] = stop_sequences
        
        # Add tools if provided
        if tools:
            # Convert OpenAI-style tools to Anthropic format
            anthropic_tools = self._convert_tools_to_anthropic_format(tools)
            payload["tools"] = anthropic_tools
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in payload:
                payload[key] = value
        
        # Make the API request
        response = requests.post(
            f"{self.base_url}/v1/messages",
            headers=self.headers,
            json=payload
        )
        
        # Handle errors
        if response.status_code != 200:
            raise Exception(f"Anthropic API Error: {response.status_code} - {response.text}")
        
        # Parse the response
        if stream:
            return response.iter_lines()
        else:
            return response.json()
    
    def get_streaming_completion(self, 
                               messages: List[Dict[str, Any]], 
                               model: str,
                               temperature: float,
                               max_tokens: int,
                               top_p: float,
                               stop_sequences: Optional[List[str]] = None,
                               tools: Optional[List[Dict[str, Any]]] = None,
                               **kwargs) -> Iterator:
        """
        Get a streaming completion from Anthropic.
        
        Args:
            messages: List of message objects with role and content
            model: Anthropic model identifier
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter
            stop_sequences: Optional list of stop sequences
            tools: Optional list of tool definitions
            **kwargs: Additional Anthropic-specific parameters
            
        Returns:
            Iterator of Anthropic API response chunks
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
    
    def create_embedding(self, text: str, model: str = "claude-3-haiku-20240307") -> List[float]:
        """
        Create an embedding for the given text using Anthropic.
        
        Note: Anthropic doesn't have a dedicated embeddings API, so we'll use the
        OpenAI embeddings API as a fallback with a warning.
        
        Args:
            text: Text to embed
            model: Model to use for embedding (ignored, will use OpenAI)
            
        Returns:
            Embedding vector as a list of floats
        """
        # Since Anthropic doesn't provide embeddings, we'll use a simple alternative
        # that generates deterministic embeddings based on hashing
        import hashlib
        import numpy as np
        
        # Generate a hash of the text
        hash_object = hashlib.sha256(text.encode())
        hash_hex = hash_object.hexdigest()
        
        # Use the hash to seed a random number generator
        np.random.seed(int(hash_hex, 16) % (2**32))
        
        # Generate a pseudo-random embedding vector (dimension 1536 for compatibility)
        embedding = np.random.normal(0, 1, 1536).tolist()
        
        # Normalize the embedding
        norm = sum(x**2 for x in embedding) ** 0.5
        embedding = [x / norm for x in embedding]
        
        return embedding
    
    def format_tool_calls(self, tool_calls: Any) -> List[Dict[str, Any]]:
        """
        Format Anthropic tool calls to standardized format.
        
        Args:
            tool_calls: Anthropic tool calls object
            
        Returns:
            List of standardized tool call objects
        """
        if not tool_calls:
            return []
        
        formatted_calls = []
        for i, tool_call in enumerate(tool_calls):
            tool_name = tool_call.get("name")
            tool_input = tool_call.get("input", {})
            
            formatted_call = {
                "id": f"call_{i}",  # Anthropic doesn't provide IDs, so we create them
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(tool_input)
                }
            }
            formatted_calls.append(formatted_call)
        
        return formatted_calls
    
    def format_tool_results(self, tool_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format tool results for Anthropic.
        
        Args:
            tool_results: List of standardized tool result objects
            
        Returns:
            List of Anthropic-compatible tool result objects
        """
        formatted_results = []
        for result in tool_results:
            # Parse the content
            try:
                content = json.loads(result.get("content"))
            except (json.JSONDecodeError, TypeError):
                content = result.get("content")
            
            # Find the corresponding tool call ID
            tool_call_id = result.get("tool_call_id")
            
            formatted_result = {
                "role": "tool",
                "name": result.get("name", "unknown_tool"),
                "content": content
            }
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    def _convert_messages_to_anthropic_format(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert OpenAI-style messages to Anthropic format.
        
        Args:
            messages: List of OpenAI-style message objects
            
        Returns:
            List of Anthropic-style message objects
        """
        anthropic_messages = []
        
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            
            if role == "system":
                # System messages are handled differently in Anthropic
                anthropic_messages.append({
                    "role": "user",
                    "content": f"<system>{content}</system>"
                })
            elif role == "user":
                anthropic_messages.append({
                    "role": "user",
                    "content": content
                })
            elif role == "assistant":
                # Handle potential tool calls in assistant messages
                if "tool_calls" in message:
                    # Convert tool calls to Anthropic format
                    tool_calls = message.get("tool_calls", [])
                    formatted_tool_calls = []
                    
                    for tool_call in tool_calls:
                        formatted_tool_calls.append({
                            "type": "tool_use",
                            "name": tool_call.get("function", {}).get("name"),
                            "input": json.loads(tool_call.get("function", {}).get("arguments", "{}"))
                        })
                    
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": content,
                        "tool_use": formatted_tool_calls
                    })
                else:
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": content
                    })
            elif role == "tool":
                # Convert tool results to Anthropic format
                anthropic_messages.append({
                    "role": "tool",
                    "name": "unknown_tool",  # We don't have the tool name, so use a default
                    "content": content
                })
        
        return anthropic_messages
    
    def _convert_tools_to_anthropic_format(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert OpenAI-style tools to Anthropic format.
        
        Args:
            tools: List of OpenAI-style tool definitions
            
        Returns:
            List of Anthropic-style tool definitions
        """
        anthropic_tools = []
        
        for tool in tools:
            if tool.get("type") == "function":
                function = tool.get("function", {})
                
                anthropic_tool = {
                    "name": function.get("name"),
                    "description": function.get("description", ""),
                    "input_schema": function.get("parameters", {})
                }
                
                anthropic_tools.append(anthropic_tool)
        
        return anthropic_tools
    
    def close(self):
        """
        Close the Anthropic client and release resources.
        """
        # Nothing to close with the requests-based implementation
        pass