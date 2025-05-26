"""
Azure AI Foundry provider implementation for YAMLLM.

This module provides an implementation of the BaseProvider interface for Azure AI Foundry.
"""

from typing import Dict, List, Any, Optional
import json
import logging
from azure.ai.inference import InferenceClient
from azure.identity import DefaultAzureCredential
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live

from yamllm.providers.base import BaseProvider, Message, ToolDefinition


class AzureFoundryProvider(BaseProvider):
    """
    Azure AI Foundry provider implementation.
    
    This class implements the BaseProvider interface for Azure AI Foundry,
    allowing access to AI models deployed in Azure AI Foundry projects.
    """
    
    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None, **kwargs):
        """
        Initialize the Azure AI Foundry provider.
        
        Args:
            api_key (str): The API key for Azure AI Foundry (or "default" to use DefaultAzureCredential).
            model (str): The model deployment name in Azure AI Foundry.
            base_url (str): The Azure AI project endpoint URL.
            **kwargs: Additional provider-specific parameters.
                project_id (str): The Azure AI project ID (optional if included in endpoint).
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.project_id = kwargs.get('project_id')
        
        # Initialize Azure AI Foundry client
        if self.api_key.lower() == "default":
            # Use DefaultAzureCredential for authentication
            credential = DefaultAzureCredential()
            self.client = InferenceClient(endpoint=self.base_url, credential=credential)
        else:
            # Use API key for authentication
            self.client = InferenceClient(endpoint=self.base_url, api_key=self.api_key)
        
        # Store additional parameters
        self.logger = kwargs.get('logger')
    
    def prepare_completion_params(self, messages: List[Message], temperature: float, max_tokens: int, 
                                 top_p: float, stop_sequences: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Prepare completion parameters for Azure AI Foundry's API.
        
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
            "deployment_id": self.model,
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
        Handle streaming response from Azure AI Foundry.
        
        Args:
            messages (List[Message]): The messages sent to the model.
            params (Dict[str, Any]): The parameters for the API request.
            
        Returns:
            str: The concatenated response text.
        """
        try:
            # Enable streaming
            params["stream"] = True
            
            # Azure AI Inference client uses different parameter names
            deployment_id = params.pop("deployment_id")
            
            response = self.client.chat_completions.create_stream(
                deployment_name=deployment_id,
                **params
            )
            
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
        Handle non-streaming response from Azure AI Foundry.
        
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
            
            # Azure AI Inference client uses different parameter names
            deployment_id = params.pop("deployment_id")
            
            response = self.client.chat_completions.create(
                deployment_name=deployment_id,
                **params
            )
            
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
        Handle streaming with tool detection for Azure AI Foundry.
        
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
            
            # Azure AI Inference client uses different parameter names
            deployment_id = preview_params.pop("deployment_id")
            
            preview_response = self.client.chat_completions.create(
                deployment_name=deployment_id,
                **preview_params
            )
            
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
        Process tool calls from Azure AI Foundry.
        
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
                response = self.client.chat_completions.create(
                    deployment_name=self.model,
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
        Create an embedding for the given text using Azure AI Foundry's API.
        
        Args:
            text (str): The text to create an embedding for.
            
        Returns:
            bytes: The embedding as bytes.
        """
        try:
            # Use the embedding deployment name if provided, otherwise use a default
            embedding_deployment = getattr(self, 'embedding_deployment', 'text-embedding-ada-002')
            
            response = self.client.embeddings.create(
                deployment_name=embedding_deployment,
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