"""
OpenAI provider implementation for YAMLLM.

This module provides an implementation of the BaseProvider interface for OpenAI.
"""

from typing import Dict, List, Any, Optional
import json
from openai import OpenAI
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live

from yamllm.providers.base import BaseProvider, Message, ToolDefinition, ToolCall


class OpenAIProvider(BaseProvider):
    """
    OpenAI provider implementation.
    
    This class implements the BaseProvider interface for OpenAI.
    """
    
    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None, **kwargs):
        """
        Initialize the OpenAI provider.
        
        Args:
            api_key (str): The API key for OpenAI.
            model (str): The model to use.
            base_url (str, optional): The base URL for the OpenAI API.
            **kwargs: Additional provider-specific parameters.
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # Initialize OpenAI client for embeddings
        self.embedding_client = OpenAI(
            api_key=api_key
        )
        
        # Store additional parameters
        self.logger = kwargs.get('logger')
    
    def prepare_completion_params(self, messages: List[Message], temperature: float, max_tokens: int, 
                                 top_p: float, stop_sequences: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Prepare completion parameters for OpenAI's API.
        
        Args:
            messages (List[Message]): The messages to send to the model.
            temperature (float): The temperature parameter for the model.
            max_tokens (int): The maximum number of tokens to generate.
            top_p (float): The top_p parameter for the model.
            stop_sequences (List[str], optional): Sequences that will stop generation.
            
        Returns:
            Dict[str, Any]: The parameters for the API request.
        """
        # Convert Message objects to dictionaries
        message_dicts = [message.to_dict() for message in messages]
        
        params = {
            "model": self.model,
            "messages": message_dicts,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        
        # Only add stop parameter if it contains actual stop sequences
        if stop_sequences and len(stop_sequences) > 0:
            params["stop"] = stop_sequences
            
        return params
    
    def handle_streaming_response(self, messages: List[Message], params: Dict[str, Any]) -> str:
        """
        Handle streaming response from OpenAI.
        
        Args:
            messages (List[Message]): The messages sent to the model.
            params (Dict[str, Any]): The parameters for the API request.
            
        Returns:
            str: The concatenated response text.
        """
        try:
            # Enable streaming
            params["stream"] = True
            
            response = self.client.chat.completions.create(**params)
            
            console = Console()
            response_text = ""
            print()
            
            with Live(console=console, refresh_per_second=10) as live:
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        response_text += chunk.choices[0].delta.content
                        md = Markdown(f"\nAI: {response_text}", style="green")
                        live.update(md)
            
            return response_text
        except Exception as e:
            if self.logger:
                self.logger.error(f"Streaming error: {str(e)}")
            raise Exception(f"Error getting streaming response: {str(e)}")
    
    def handle_non_streaming_response(self, messages: List[Message], params: Dict[str, Any], 
                                     tools: Optional[List[ToolDefinition]] = None) -> str:
        """
        Handle non-streaming response from OpenAI.
        
        Args:
            messages (List[Message]): The messages sent to the model.
            params (Dict[str, Any]): The parameters for the API request.
            tools (List[ToolDefinition], optional): Tool definitions.
            
        Returns:
            str: The response text.
        """
        try:
            # Add tools if available
            if tools:
                params["tools"] = [tool.to_dict() for tool in tools]
                params["tool_choice"] = "auto"
            
            response = self.client.chat.completions.create(**params)
            
            # Check if the model wants to use a tool
            if tools and hasattr(response.choices[0].message, "tool_calls") and response.choices[0].message.tool_calls:
                # Return the model message for tool processing
                return response.choices[0].message
            else:
                response_text = response.choices[0].message.content
                
                # Display the response
                console = Console()
                if any(marker in response_text for marker in ['###', '```', '*', '_', '-']):
                    md = Markdown("\nAI:" + response_text, style="green")
                    console.print(md)
                else:
                    console.print("\nAI:" + response_text, style="green")
                    
                return response_text
        except Exception as e:
            if self.logger:
                self.logger.error(f"Non-streaming error: {str(e)}")
            raise Exception(f"Error getting non-streaming response: {str(e)}")
    
    def handle_streaming_with_tool_detection(self, messages: List[Message], params: Dict[str, Any], 
                                           tools: Optional[List[ToolDefinition]] = None) -> str:
        """
        Handle streaming with tool detection for OpenAI.
        
        Args:
            messages (List[Message]): The messages sent to the model.
            params (Dict[str, Any]): The parameters for the API request.
            tools (List[ToolDefinition], optional): Tool definitions.
            
        Returns:
            str: The response text.
        """
        try:
            # Make a low-token request to see if the model will use tools
            preview_params = params.copy()
            preview_params["max_tokens"] = 10  # Just enough to detect tool usage
            
            if tools:
                preview_params["tools"] = [tool.to_dict() for tool in tools]
                preview_params["tool_choice"] = "auto"
                
            preview_response = self.client.chat.completions.create(**preview_params)
            
            # Check if model wants to use tools
            if (tools and hasattr(preview_response.choices[0].message, "tool_calls") 
                and preview_response.choices[0].message.tool_calls):
                # Model wants to use tools, use non-streaming
                console = Console()
                console.print("\n[yellow]Using tools to answer this question...[/yellow]")
                return self.handle_non_streaming_response(messages, params, tools)
            else:
                # Model doesn't need tools, use streaming
                return self.handle_streaming_response(messages, params)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Preview request failed: {str(e)}")
            # Fall back to streaming without tool detection
            return self.handle_streaming_response(messages, params)
    
    def process_tool_calls(self, messages: List[Message], model_message: Any, 
                          execute_tool_func: callable, max_iterations: int = 5) -> str:
        """
        Process tool calls from OpenAI.
        
        Args:
            messages (List[Message]): The messages sent to the model.
            model_message (Any): The message from the model containing tool calls.
            execute_tool_func (callable): Function to execute a tool.
            max_iterations (int, optional): Maximum number of tool call iterations.
            
        Returns:
            str: The final response text.
        """
        console = Console()
        iteration = 0
        current_messages = [message.to_dict() for message in messages]
        
        while iteration < max_iterations:
            iteration += 1
            
            # Add the model's message to the conversation
            current_messages.append({
                "role": "assistant",
                "content": model_message.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    } for tool_call in model_message.tool_calls
                ]
            })
            
            # Process each tool call
            for tool_call in model_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                # Display tool call information
                console.print("\n[bold yellow]Tool Call Requested:[/bold yellow]")
                console.print(f"[yellow]Function:[/yellow] {tool_name}")
                console.print(f"[yellow]Arguments:[/yellow] {json.dumps(tool_args)}")
                
                # Execute the tool
                try:
                    tool_result = execute_tool_func(tool_name, tool_args)
                    
                    # Display tool result
                    console.print("\n[bold green]Tool Result:[/bold green]")
                    if isinstance(tool_result, str) and any(marker in tool_result for marker in ['###', '```', '*', '_', '-']):
                        md = Markdown(tool_result, style="green")
                        console.print(md)
                    else:
                        console.print(str(tool_result), style="green")
                    
                    # Add the tool result to the conversation
                    current_messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_name,
                        "content": str(tool_result)
                    })
                except Exception as e:
                    error_message = f"Error executing tool {tool_name}: {str(e)}"
                    if self.logger:
                        self.logger.error(error_message)
                    
                    # Add the error message to the conversation
                    current_messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_name,
                        "content": error_message
                    })
            
            # Get the model's response to the tool results
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=current_messages
                )
                
                model_message = response.choices[0].message
                
                # Check if the model wants to use more tools
                if hasattr(model_message, "tool_calls") and model_message.tool_calls:
                    # Continue the loop with the new tool calls
                    continue
                else:
                    # Model has finished using tools, return the final response
                    response_text = model_message.content
                    
                    # Display the final response
                    console.print("\n[bold blue]Final Response:[/bold blue]")
                    if any(marker in response_text for marker in ['###', '```', '*', '_', '-']):
                        md = Markdown(response_text, style="green")
                        console.print(md)
                    else:
                        console.print(response_text, style="green")
                    
                    return response_text
            except Exception as e:
                error_message = f"Error getting model response: {str(e)}"
                if self.logger:
                    self.logger.error(error_message)
                raise Exception(error_message)
        
        # If we've reached the maximum number of iterations, return the last model message
        if hasattr(model_message, "content") and model_message.content:
            return model_message.content
        else:
            return "Maximum tool call iterations reached without a final response."
    
    def create_embedding(self, text: str) -> bytes:
        """
        Create an embedding for the given text using OpenAI's API.
        
        Args:
            text (str): The text to create an embedding for.
            
        Returns:
            bytes: The embedding as bytes.
        """
        try:
            response = self.embedding_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            
            # Convert the embedding to bytes
            import numpy as np
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            return embedding.tobytes()
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating embedding: {str(e)}")
            raise Exception(f"Error creating embedding: {str(e)}")
    
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
            if hasattr(tool_call, 'function') and hasattr(tool_call, 'id'):
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
                    "id": tool_call.get("id", f"call_{len(formatted_calls)}"),
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