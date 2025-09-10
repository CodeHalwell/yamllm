"""
MistralAI provider implementation for YAMLLM.

This module provides an implementation of the BaseProvider interface for MistralAI
using the official mistralai Python SDK.
"""

from typing import Dict, List, Any, Optional
import json
import numpy as np
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live

from mistralai.sdk import Mistral
try:
    from mistralai.models.chat_completion import ChatCompletionResponse
except ImportError:
    # Handle newer mistralai SDK versions
    try:
        from mistralai.models import ChatCompletionResponse
    except ImportError:
        # Fallback for compatibility
        ChatCompletionResponse = object

from yamllm.providers.base import BaseProvider, Message, ToolDefinition, ToolCall


class MistralProvider(BaseProvider):
    """
    MistralAI provider implementation.
    
    This class implements the BaseProvider interface for MistralAI
    using the official mistralai Python SDK.
    """
    
    # No longer using real-time query keywords - model will determine when to use tools
    
    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None, **kwargs):
        """
        Initialize the MistralAI provider.
        
        Args:
            api_key (str): The API key for MistralAI.
            model (str): The model to use.
            base_url (str, optional): The base URL for the MistralAI API.
            **kwargs: Additional provider-specific parameters.
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        
        # Initialize Mistral client
        self.client = Mistral(
            api_key=self.api_key,
            server_url=self.base_url
        )
        
        # Store additional parameters
        self.logger = kwargs.get('logger')
        
    def prepare_completion_params(self, messages: List[Message], temperature: float, max_tokens: int, 
                                 top_p: float, stop_sequences: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Prepare completion parameters for Mistral's API.
        
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
        Handle streaming response from Mistral.
        
        Args:
            messages (List[Message]): The messages sent to the model.
            params (Dict[str, Any]): The parameters for the API request.
            
        Returns:
            str: The concatenated response text.
        """
        try:
            # Enable streaming
            params["stream"] = True
            
            # Use the chat.complete method with streaming
            response_stream = self.client.chat.complete(**params)
            
            console = Console()
            response_text = ""
            print()
            
            with Live(console=console, refresh_per_second=10) as live:
                for chunk in response_stream:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        delta_content = chunk.choices[0].delta.content
                        response_text += delta_content
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
        Handle non-streaming response from Mistral.
        
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
                formatted_tools = []
                for tool in tools:
                    formatted_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters
                        }
                    })
                params["tools"] = formatted_tools
                params["tool_choice"] = "auto"
            
            # Use the chat.complete method
            response = self.client.chat.complete(**params)
            
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
        Handle streaming with tool detection for Mistral.
        
        Args:
            messages (List[Message]): The messages sent to the model.
            params (Dict[str, Any]): The parameters for the API request.
            tools (List[ToolDefinition], optional): Tool definitions.
            
        Returns:
            str: The response text.
        """
        try:
            # For Mistral, we don't do a preview request as it might not support it well
            # Instead, we check if tools are provided and use non-streaming if they are
            if tools:
                console = Console()
                console.print("\n[yellow]Using non-streaming mode for tool support...[/yellow]")
                return self.handle_non_streaming_response(messages, params, tools)
            else:
                return self.handle_streaming_response(messages, params)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error in streaming with tool detection: {str(e)}")
            # Fall back to streaming without tool detection
            return self.handle_streaming_response(messages, params)
    
    def process_tool_calls(self, messages: List[Message], model_message: Any, 
                          execute_tool_func: callable, max_iterations: int = 5) -> str:
        """
        Process tool calls from Mistral.
        
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
                response = self.client.chat.complete(
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
        Create an embedding for the given text using Mistral's embeddings API.
        
        Args:
            text (str): The text to create an embedding for.
            
        Returns:
            bytes: The embedding as bytes.
        """
        try:
            # Use the embeddings.create method with mistral-embed model
            embedding_model = "mistral-embed"
            
            response = self.client.embeddings.create(
                model=embedding_model,
                inputs=text
            )
            
            # Convert the embedding to bytes
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            return embedding.tobytes()
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating embedding: {str(e)}")
            raise Exception(f"Error creating embedding: {str(e)}")
            
    def close(self):
        """
        Close the client and release resources.
        """
        # Using mistralai's utils function to close clients
        from mistralai import close_clients
        
    def format_tool_calls(self, tool_calls: Any) -> List[Dict[str, Any]]:
        """
        Format Mistral tool calls to standardized format.
        
        Args:
            tool_calls: Mistral tool calls object
            
        Returns:
            List of standardized tool call objects
        """
        if not tool_calls:
            return []
        
        formatted_calls = []
        for tool_call in tool_calls:
            # Mistral uses OpenAI-compatible format for tool calls
            if hasattr(tool_call, 'function') and hasattr(tool_call, 'id'):
                formatted_call = {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                }
            elif isinstance(tool_call, dict):
                # Handle dictionary format
                formatted_call = {
                    "id": tool_call.get("id", f"call_{len(formatted_calls)}"),
                    "type": "function",
                    "function": {
                        "name": tool_call.get("function", {}).get("name"),
                        "arguments": tool_call.get("function", {}).get("arguments")
                    }
                }
            else:
                # Skip unrecognized formats
                continue
            
            formatted_calls.append(formatted_call)
        
        return formatted_calls
    
    def format_tool_results(self, tool_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format tool results for Mistral.
        
        Args:
            tool_results: List of standardized tool result objects
            
        Returns:
            List of Mistral-compatible tool result objects
        """
        formatted_results = []
        for result in tool_results:
            formatted_result = {
                "tool_call_id": result.get("tool_call_id"),
                "role": "tool",
                "name": result.get("name"),
                "content": result.get("content")
            }
            formatted_results.append(formatted_result)
        
        return formatted_results