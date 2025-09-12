"""
Anthropic provider implementation for YAMLLM.

This module implements the BaseProvider interface for Anthropic's API,
supporting Claude models and tools using the official Anthropic Python SDK.
"""

import json
from typing import Dict, List, Any, Optional, Iterator

from anthropic import Anthropic
from yamllm.providers.exceptions import ProviderError

from yamllm.providers.base import BaseProvider


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
        
        # Initialize the Anthropic client
        self.client = Anthropic(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # Store additional parameters
        self.timeout = kwargs.get("timeout", 600)
    
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
        
        # Prepare the request parameters
        params = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }
        
        # Add optional parameters if provided
        if stop_sequences:
            params["stop_sequences"] = stop_sequences
        
        # Add tools if provided
        if tools:
            # Convert OpenAI-style tools to Anthropic format
            anthropic_tools = self._convert_tools_to_anthropic_format(tools)
            params["tools"] = anthropic_tools
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in params:
                params[key] = value
        
        try:
            response = self.client.messages.create(**params)
            if stream:
                return response
            return response.model_dump()
        except Exception as e:
            raise ProviderError("Anthropic", f"API error: {str(e)}", original_error=e) from e
    
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
        # Convert OpenAI-style messages to Anthropic format
        anthropic_messages = self._convert_messages_to_anthropic_format(messages)
        
        # Prepare the request parameters
        params = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        
        # Add optional parameters if provided
        if stop_sequences:
            params["stop_sequences"] = stop_sequences
        
        # Add tools if provided
        if tools:
            # Convert OpenAI-style tools to Anthropic format
            anthropic_tools = self._convert_tools_to_anthropic_format(tools)
            params["tools"] = anthropic_tools
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in params:
                params[key] = value
        
        try:
            # Use the Anthropic SDK's stream method
            stream = self.client.messages.stream(**params)
            return stream
        except Exception as e:
            raise ProviderError("Anthropic", f"Streaming error: {str(e)}", original_error=e)
    
    def create_embedding(self, text: str, model: str = "claude-3-haiku-20240307") -> List[float]:
        """
        Anthropic does not provide an embeddings API.
        Raise a ProviderError so callers can fall back to another provider.
        """
        raise ProviderError("Anthropic", "Embeddings not supported by Anthropic")
    
    def format_tool_calls(self, tool_calls: Any) -> List[Dict[str, Any]]:
        """
        Format Anthropic tool calls to standardized format.
        
        Args:
            tool_calls: Anthropic tool calls object from SDK response
            
        Returns:
            List of standardized tool call objects
        """
        if not tool_calls:
            return []
        
        formatted_calls = []
        for i, tool_call in enumerate(tool_calls):
            # Handle SDK's response format
            if hasattr(tool_call, 'model_dump'):
                tool_call = tool_call.model_dump()
            
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
                # Using the Anthropic SDK's convention for system messages
                anthropic_messages.append({
                    "role": "user",
                    "content": f"<s>{content}</s>"
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
                    "name": message.get("name", "unknown_tool"),
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
        if hasattr(self, 'client') and self.client:
            self.client.close()
