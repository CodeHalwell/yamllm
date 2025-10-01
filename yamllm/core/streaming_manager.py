"""
Streaming management component for YAMLLM.

This module handles all streaming-related functionality for LLM responses.
"""

from typing import Dict, List, Any, Optional, Callable
import logging
import json


class StreamingManager:
    """
    Manages streaming responses from LLM providers.
    
    This class extracts streaming logic from the main LLM class
    for better separation of concerns.
    """
    
    def __init__(
        self,
        provider_client: Any,
        provider_name: str,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop_sequences: Optional[List[str]] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize streaming manager.
        
        Args:
            provider_client: The provider client instance
            provider_name: Name of the provider
            model: Model name
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            stop_sequences: Optional stop sequences
            stream_callback: Callback for streaming text chunks
            event_callback: Callback for events
            logger: Optional logger instance
        """
        self.provider_client = provider_client
        self.provider_name = provider_name
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.stop_sequences = stop_sequences or []
        self.stream_callback = stream_callback
        self.event_callback = event_callback
        self.logger = logger or logging.getLogger('yamllm.streaming')
    
    def handle_streaming_response(
        self,
        messages: List[Dict[str, Any]]
    ) -> str:
        """
        Handle streaming response without tools.
        
        Args:
            messages: List of messages to send
            
        Returns:
            The complete response text
        """
        accumulated = ""
        
        try:
            stream = self.provider_client.get_completion_streaming(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                stop=self.stop_sequences if self.stop_sequences else None
            )
            
            for chunk in stream:
                delta = self._extract_chunk_text(chunk)
                if delta:
                    accumulated += delta
                    if self.stream_callback:
                        self.stream_callback(delta)
            
            return accumulated
            
        except Exception as e:
            self.logger.error(f"Error in streaming response: {e}")
            raise
    
    def handle_streaming_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_executor: Callable[[str, Dict[str, Any]], Any]
    ) -> str:
        """
        Handle streaming response with tool support.
        
        Args:
            messages: List of messages to send
            tools: Tool definitions
            tool_executor: Function to execute tools
            
        Returns:
            The final response text
        """
        accumulated = ""
        tool_calls = []
        
        try:
            stream = self.provider_client.get_completion_streaming(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                stop=self.stop_sequences if self.stop_sequences else None,
                tools=tools
            )
            
            for chunk in stream:
                # Handle text content
                delta = self._extract_chunk_text(chunk)
                if delta:
                    accumulated += delta
                    if self.stream_callback:
                        self.stream_callback(delta)
                
                # Handle tool calls
                tool_call = self._extract_tool_call(chunk)
                if tool_call:
                    tool_calls.append(tool_call)
            
            # Execute tool calls if any
            if tool_calls:
                for tool_call in tool_calls:
                    try:
                        tool_name = tool_call.get('name', '')
                        tool_args = tool_call.get('arguments', {})
                        
                        if isinstance(tool_args, str):
                            tool_args = json.loads(tool_args)
                        
                        if self.event_callback:
                            self.event_callback({
                                'type': 'tool_call',
                                'tool': tool_name,
                                'arguments': tool_args
                            })
                        
                        # Execute tool
                        result = tool_executor(tool_name, tool_args)
                        
                        if self.event_callback:
                            self.event_callback({
                                'type': 'tool_result',
                                'tool': tool_name,
                                'result': result
                            })
                        
                        # Add tool result to messages and get follow-up
                        messages.append({
                            'role': 'assistant',
                            'tool_calls': [tool_call]
                        })
                        messages.append({
                            'role': 'tool',
                            'tool_call_id': tool_call.get('id', ''),
                            'content': str(result)
                        })
                        
                        # Get follow-up response
                        follow_up = self.handle_streaming_response(messages)
                        accumulated += follow_up
                        
                    except Exception as e:
                        self.logger.error(f"Error executing tool {tool_name}: {e}")
                        error_msg = f"\n[Tool error: {e}]\n"
                        accumulated += error_msg
                        if self.stream_callback:
                            self.stream_callback(error_msg)
            
            return accumulated
            
        except Exception as e:
            self.logger.error(f"Error in streaming with tools: {e}")
            raise
    
    def _extract_chunk_text(self, chunk) -> str:
        """Extract text from a streaming chunk."""
        try:
            if self.provider_name.lower() in ("openai", "azure_openai", "mistral", "openrouter", "deepseek"):
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        return delta.content
            elif self.provider_name.lower() == "anthropic":
                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                    return chunk.delta.text
            elif self.provider_name.lower() == "google":
                if hasattr(chunk, 'text'):
                    return chunk.text
        except Exception as e:
            self.logger.debug(f"Error extracting chunk text: {e}")
        
        return ""
    
    def _extract_tool_call(self, chunk) -> Optional[Dict[str, Any]]:
        """Extract tool call from a streaming chunk."""
        try:
            if self.provider_name.lower() in ("openai", "azure_openai", "mistral", "openrouter", "deepseek"):
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'tool_calls') and delta.tool_calls:
                        tool_call = delta.tool_calls[0]
                        return {
                            'id': getattr(tool_call, 'id', ''),
                            'name': getattr(tool_call.function, 'name', ''),
                            'arguments': getattr(tool_call.function, 'arguments', '{}')
                        }
        except Exception as e:
            self.logger.debug(f"Error extracting tool call: {e}")
        
        return None
