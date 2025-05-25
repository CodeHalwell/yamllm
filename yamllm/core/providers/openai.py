"""
OpenAI provider implementation for YAMLLM.

This module implements the BaseProvider interface for OpenAI's API,
supporting the latest features including the response API and tool calling.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Iterator, Callable

from openai import OpenAI, OpenAIError
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

from yamllm.core.providers.base import BaseProvider


logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """
    OpenAI provider implementation using the latest OpenAI API.
    
    This class implements the BaseProvider interface for OpenAI models,
    supporting chat completions, embeddings, tool calling, and the Responses API pattern
    for interactive tool usage.
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
        
        try:
            # Make the API request
            return self.client.chat.completions.create(**params)
        except OpenAIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            # Re-raise with more context
            raise OpenAIError(f"Error calling OpenAI API: {str(e)}")
    
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
        try:
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
        except OpenAIError as e:
            logger.error(f"OpenAI API streaming error: {str(e)}")
            # Re-raise with more context
            raise OpenAIError(f"Error setting up streaming with OpenAI API: {str(e)}")
    
    def process_tool_calls(self,
                           messages: List[Dict[str, Any]],
                           model: str,
                           temperature: float,
                           max_tokens: int,
                           top_p: float,
                           tools: List[Dict[str, Any]],
                           tool_executor: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]],
                           stop_sequences: Optional[List[str]] = None,
                           stream: bool = False,
                           max_iterations: int = 10,
                           **kwargs) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """
        Process tool calls using the Responses API pattern.
        
        This method implements the interactive tool usage pattern, where the model
        can call tools, receive results, and continue the conversation based on
        those results.
        
        Args:
            messages: List of message objects with role and content
            model: OpenAI model identifier
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter
            tools: List of tool definitions
            tool_executor: Function to execute tool calls and return results
            stop_sequences: Optional list of stop sequences
            stream: Whether to stream the final response
            max_iterations: Maximum number of tool call iterations
            **kwargs: Additional OpenAI-specific parameters
            
        Returns:
            OpenAI ChatCompletion object or streaming iterator
        """
        current_messages = messages.copy()
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            
            try:
                # Request completion with tools
                response = self.get_completion(
                    messages=current_messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stop_sequences=stop_sequences,
                    tools=tools,
                    stream=False,  # Always use non-streaming for tool calls
                    **kwargs
                )
                
                # Get the assistant message
                assistant_message = response.choices[0].message
                
                # Add the assistant message to the conversation
                current_messages.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    **({"tool_calls": assistant_message.tool_calls} if assistant_message.tool_calls else {})
                })
                
                # Check if there are tool calls to process
                if not assistant_message.tool_calls:
                    # No tool calls, return the final response
                    if stream:
                        # If streaming was requested, return a streaming response for the final result
                        return self.get_streaming_completion(
                            messages=current_messages,
                            model=model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            top_p=top_p,
                            stop_sequences=stop_sequences,
                            tools=tools,
                            **kwargs
                        )
                    return response
                
                # Format tool calls for execution
                formatted_tool_calls = self.format_tool_calls(assistant_message.tool_calls)
                
                # Execute the tools
                tool_results = tool_executor(formatted_tool_calls)
                
                # Format the results for OpenAI
                formatted_results = self.format_tool_results(tool_results)
                
                # Add the tool results to the conversation
                current_messages.extend(formatted_results)
                
            except OpenAIError as e:
                logger.error(f"OpenAI API error during tool processing: {str(e)}")
                raise OpenAIError(f"Error processing tool calls with OpenAI API: {str(e)}")
        
        # If we reach here, we've hit the maximum number of iterations
        logger.warning(f"Reached maximum tool call iterations ({max_iterations})")
        
        # Return the final response
        if stream:
            return self.get_streaming_completion(
                messages=current_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop_sequences=stop_sequences,
                **kwargs
            )
        
        return self.get_completion(
            messages=current_messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop_sequences=stop_sequences,
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
        try:
            response = self.embedding_client.embeddings.create(
                input=text,
                model=model
            )
            
            return response.data[0].embedding
        except OpenAIError as e:
            logger.error(f"OpenAI API embedding error: {str(e)}")
            raise OpenAIError(f"Error creating embeddings with OpenAI API: {str(e)}")
    
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
            # Handle both ChatCompletionMessageToolCall objects and raw dict formats
            if isinstance(tool_call, ChatCompletionMessageToolCall):
                formatted_call = {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                }
            else:
                # Assume it's already a dict-like structure
                formatted_call = {
                    "id": tool_call.get("id"),
                    "type": "function",
                    "function": {
                        "name": tool_call.get("function", {}).get("name"),
                        "arguments": tool_call.get("function", {}).get("arguments")
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