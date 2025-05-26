"""
Google Gemini provider implementation for YAMLLM.

This module provides an implementation of the BaseProvider interface for Google Gemini.
"""

from typing import Dict, List, Any, Optional
import json
import google.generativeai as genai
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live

from yamllm.providers.base import BaseProvider, Message, ToolDefinition
from yamllm.tools.utility_tools import WebSearch


class GoogleGeminiProvider(BaseProvider):
    """
    Google Gemini provider implementation.
    
    This class implements the BaseProvider interface for Google Gemini.
    """
    
    # Real-time query keywords
    real_time_keywords = [
        # Weather and natural phenomena
        "weather", "forecast", "temperature", "humidity", "precipitation", "rain", "snow", "storm", 
        "hurricane", "tornado", "earthquake", "tsunami", "typhoon", "cyclone", "flood", "drought", 
        "wildfire", "air quality", "pollen", "uv index", "sunrise", "sunset", "climate",
        
        # News and current events
        "news", "headline", "latest", "breaking", "current", "recent", "today", "yesterday",
        "this week", "this month", "ongoing", "developing", "situation", "event", "incident", 
        "announcement", "press release", "update", "coverage", "report", "bulletin", "fixture",
        
        # Sports and entertainment
        "score", "game", "match", "tournament", "championship", "playoff", "standings", 
        "leaderboard", "box office", "premiere", "release", "concert", "performance", 
        "episode", "ratings", "award", "nominations", "season", "show", "event",
        
        # Time-specific queries
        "now", "currently", "present", "moment", "tonight", "this morning", "this afternoon", 
        "this evening", "upcoming", "soon", "shortly", "imminent", "expected", "anticipated", 
        "scheduled", "real-time", "live", "happening", "occurring", "next"
    ]
    
    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None, **kwargs):
        """
        Initialize the Google Gemini provider.
        
        Args:
            api_key (str): The API key for Google Gemini.
            model (str): The model to use.
            base_url (str, optional): The base URL for the Google Gemini API.
            **kwargs: Additional provider-specific parameters.
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        
        # Configure the Google GenAI client
        genai.configure(api_key=self.api_key)
        
        # Set custom API endpoint if provided
        if self.base_url:
            genai.configure(transport="rest", client_options={"api_endpoint": self.base_url})
        
        # Initialize the model
        self.client = genai.GenerativeModel(model_name=self.model)
        
        # Store additional parameters
        self.logger = kwargs.get('logger')
        self.tools = kwargs.get('tools', [])
        self.tools_enabled = kwargs.get('tools_enabled', False)
    
    def _convert_messages_to_google_format(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """
        Convert YAMLLM messages to Google's format.
        
        Args:
            messages (List[Message]): The messages to convert.
            
        Returns:
            List[Dict[str, Any]]: The messages in Google's format.
        """
        processed_messages = []
        
        # First, look for system messages and combine them if there are multiple
        system_content = []
        for message in messages:
            if message.role == "system":
                system_content.append(message.content)
        
        # Add combined system message at the beginning if there were any
        if system_content:
            combined_system = "System instructions:\n" + "\n".join(system_content)
            processed_messages.append({
                "role": "user",
                "parts": [{"text": combined_system}]
            })
            
            # Add a model response to acknowledge the system instructions
            processed_messages.append({
                "role": "model",
                "parts": [{"text": "I'll follow these instructions."}]
            })
        
        # Process the rest of the messages
        for message in messages:
            role = message.role
            
            # Skip system messages as they're already handled
            if role == "system":
                continue
                
            elif role == "user":
                processed_messages.append({
                    "role": "user",
                    "parts": [{"text": message.content}]
                })
                
            elif role == "assistant":
                processed_messages.append({
                    "role": "model",
                    "parts": [{"text": message.content}]
                })
                
            elif role == "tool":
                # Tool messages will be handled separately in process_tool_calls
                continue
                
        return processed_messages

    def prepare_completion_params(self, messages: List[Message], temperature: float, max_tokens: int, 
                                 top_p: float, stop_sequences: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Prepare completion parameters for Google Gemini's API.
        
        Args:
            messages (List[Message]): The messages to send to the model.
            temperature (float): The temperature parameter for the model.
            max_tokens (int): The maximum number of tokens to generate.
            top_p (float): The top_p parameter for the model.
            stop_sequences (List[str], optional): Sequences that will stop generation.
            
        Returns:
            Dict[str, Any]: The parameters for the API request.
        """
        # Convert messages to Google's format
        processed_messages = self._convert_messages_to_google_format(messages)
        
        # Build generation config
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "top_p": top_p,
        }
        
        # Add stop sequences if provided
        if stop_sequences and len(stop_sequences) > 0:
            generation_config["stop_sequences"] = stop_sequences
            
        params = {
            "contents": processed_messages,
            "generation_config": generation_config
        }
            
        return params
    
    def handle_streaming_response(self, messages: List[Message], params: Dict[str, Any]) -> str:
        """
        Handle streaming response from Google Gemini.
        
        Args:
            messages (List[Message]): The messages sent to the model.
            params (Dict[str, Any]): The parameters for the API request.
            
        Returns:
            str: The concatenated response text.
        """
        try:
            console = Console()
            response_text = ""
            print()
            
            # Start streaming with the Google Gemini API
            stream = self.client.generate_content(
                contents=params["contents"],
                generation_config=params["generation_config"],
                stream=True
            )
            
            with Live(console=console, refresh_per_second=10) as live:
                for chunk in stream:
                    if hasattr(chunk, "text") and chunk.text:
                        response_text += chunk.text
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
        Handle non-streaming response from Google Gemini.
        
        Args:
            messages (List[Message]): The messages sent to the model.
            params (Dict[str, Any]): The parameters for the API request.
            tools (List[ToolDefinition], optional): Tool definitions.
            
        Returns:
            str: The response text.
        """
        try:
            # For Google Gemini, check if this is a real-time query that should use web_search
            is_real_time_query = False
            last_user_msg = next((m.content for m in messages if m.role == "user"), "")
            
            # Extract just the user's question without any context annotations
            actual_query = last_user_msg.split("\nRelevant context from previous conversations:")[0].strip()
            
            if any(keyword in actual_query.lower() for keyword in self.real_time_keywords) and "web_search" in self.tools:
                is_real_time_query = True
            
            if is_real_time_query and tools and self.tools_enabled:
                # Force web search for real-time queries
                console = Console()
                console.print("\n[yellow]Using tools to answer this real-time question...[/yellow]")
                
                # Create a web search directly instead of going through the API
                web_search = WebSearch()
                web_search_args = {
                    "query": actual_query,  # Use the clean query without context annotations
                    "max_results": 5
                }
                
                # Display tool call information
                console.print("\n[bold yellow]Tool Call Requested:[/bold yellow]")
                console.print("[yellow]Function:[/yellow] web_search")
                console.print(f"[yellow]Arguments:[/yellow] {json.dumps(web_search_args)}")
                
                # Execute the search directly
                search_results = web_search.execute(**web_search_args)
                
                # Display tool result
                console.print("\n[bold green]Tool Result:[/bold green]")
                if isinstance(search_results, str) and any(marker in search_results for marker in ['###', '```', '*', '_', '-']):
                    md = Markdown(search_results, style="green")
                    console.print(md)
                else:
                    console.print(str(search_results), style="green")
                
                # Add the search results to the messages
                search_context = f"Here are the search results for '{actual_query}':\n\n{search_results}"
                
                # Add search context to the contents
                search_system_msg = {
                    "role": "user",
                    "parts": [{"text": f"System information: {search_context}"}]
                }
                params["contents"].append(search_system_msg)
            
            # Add tools if available and not a real-time query
            if tools and not is_real_time_query:
                # Convert tools to Google's format and add them to the model
                google_tools = []
                for tool in tools:
                    google_tools.append({
                        "function_declarations": [{
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters
                        }]
                    })
                
                # Use tools with Google GenAI
                response = self.client.generate_content(
                    contents=params["contents"],
                    generation_config=params["generation_config"],
                    tools=google_tools
                )
            else:
                # Regular response without tools
                response = self.client.generate_content(
                    contents=params["contents"],
                    generation_config=params["generation_config"]
                )
            
            # Check if the model wants to use a tool
            if tools and not is_real_time_query and hasattr(response, "candidates") and \
                response.candidates[0].content.parts and \
                hasattr(response.candidates[0].content.parts[0], "function_call"):
                # Return the model message for tool processing
                return response
            else:
                # Get the text response
                response_text = response.text
                
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
        Handle streaming with tool detection for Google Gemini.
        
        Args:
            messages (List[Message]): The messages sent to the model.
            params (Dict[str, Any]): The parameters for the API request.
            tools (List[ToolDefinition], optional): Tool definitions.
            
        Returns:
            str: The response text.
        """
        try:
            # For Google Gemini, check if this is a real-time query that should use web_search
            is_real_time_query = False
            last_user_msg = next((m.content for m in messages if m.role == "user"), "")
            
            # Extract just the user's question without any context annotations
            actual_query = last_user_msg.split("\nRelevant context from previous conversations:")[0].strip()
            
            if any(keyword in actual_query.lower() for keyword in self.real_time_keywords) and "web_search" in self.tools:
                is_real_time_query = True
            
            if is_real_time_query and tools and self.tools_enabled:
                # For real-time queries, use non-streaming with web search
                return self.handle_non_streaming_response(messages, params, tools)
            
            # Make a low-token request to see if the model will use tools
            preview_params = params.copy()
            preview_params["generation_config"] = preview_params["generation_config"].copy()
            preview_params["generation_config"]["max_output_tokens"] = 10  # Just enough to detect tool usage
            
            if tools:
                # Convert tools to Google's format
                google_tools = []
                for tool in tools:
                    google_tools.append({
                        "function_declarations": [{
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters
                        }]
                    })
                
                # Make a preview request with tools
                preview_response = self.client.generate_content(
                    contents=preview_params["contents"],
                    generation_config=preview_params["generation_config"],
                    tools=google_tools
                )
                
                # Check if model wants to use tools
                if (hasattr(preview_response, "candidates") and 
                    preview_response.candidates[0].content.parts and 
                    hasattr(preview_response.candidates[0].content.parts[0], "function_call")):
                    # Model wants to use tools, use non-streaming
                    console = Console()
                    console.print("\n[yellow]Using tools to answer this question...[/yellow]")
                    return self.handle_non_streaming_response(messages, params, tools)
                else:
                    # Model doesn't need tools, use streaming
                    return self.handle_streaming_response(messages, params)
            else:
                # No tools available, use streaming
                return self.handle_streaming_response(messages, params)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Preview request failed: {str(e)}")
            # Fall back to streaming without tool detection
            return self.handle_streaming_response(messages, params)
    
    def process_tool_calls(self, messages: List[Message], model_message: Any, 
                          execute_tool_func: callable, max_iterations: int = 5) -> str:
        """
        Process tool calls from Google Gemini.
        
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
        
        # Convert messages to Google's format for the conversation history
        current_messages = self._convert_messages_to_google_format(messages)
        
        while iteration < max_iterations:
            iteration += 1
            
            # Extract function call from the model response
            if hasattr(model_message, "candidates") and model_message.candidates:
                function_calls = []
                for candidate in model_message.candidates:
                    if hasattr(candidate.content, "parts"):
                        for part in candidate.content.parts:
                            if hasattr(part, "function_call"):
                                function_calls.append(part.function_call)
            else:
                # No function calls found, break the loop
                if hasattr(model_message, "text"):
                    return model_message.text
                else:
                    return "No response from the model."
            
            # Add the model's response to the conversation
            # We'll add the function calls later after processing
            model_response_text = model_message.text if hasattr(model_message, "text") else ""
            current_messages.append({
                "role": "model",
                "parts": [{"text": model_response_text}]  # Will be updated with function calls
            })
            
            # Process each function call
            for function_call in function_calls:
                tool_name = function_call.name
                tool_args = json.loads(function_call.args)
                
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
                        "role": "user",
                        "parts": [{
                            "function_response": {
                                "name": tool_name,
                                "response": {
                                    "content": str(tool_result)
                                }
                            }
                        }]
                    })
                except Exception as e:
                    error_message = f"Error executing tool {tool_name}: {str(e)}"
                    if self.logger:
                        self.logger.error(error_message)
                    
                    # Add the error message to the conversation
                    current_messages.append({
                        "role": "user",
                        "parts": [{
                            "function_response": {
                                "name": tool_name,
                                "response": {
                                    "content": error_message
                                }
                            }
                        }]
                    })
            
            # Get the model's response to the tool results
            try:
                # Convert tools to Google's format for the next iteration
                google_tools = []
                for tool in [t for t in messages if isinstance(t, ToolDefinition)]:
                    google_tools.append({
                        "function_declarations": [{
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters
                        }]
                    })
                
                # Get response with the updated conversation
                model_message = self.client.generate_content(
                    contents=current_messages,
                    tools=google_tools
                )
                
                # Check if the model wants to use more tools
                has_function_call = False
                if hasattr(model_message, "candidates") and model_message.candidates:
                    for candidate in model_message.candidates:
                        if hasattr(candidate.content, "parts"):
                            for part in candidate.content.parts:
                                if hasattr(part, "function_call"):
                                    has_function_call = True
                                    break
                
                if has_function_call:
                    # Continue the loop with the new tool calls
                    continue
                else:
                    # Model has finished using tools, return the final response
                    response_text = model_message.text
                    
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
        if hasattr(model_message, "text"):
            return model_message.text
        else:
            return "Maximum tool call iterations reached without a final response."
    
    def create_embedding(self, text: str) -> bytes:
        """
        Create an embedding for the given text.
        
        Uses Google's embedding model to create embeddings.
        
        Args:
            text (str): The text to create an embedding for.
            
        Returns:
            bytes: The embedding as bytes.
        """
        try:
            # Use Google's embedding model
            embedding_model = genai.get_model("models/embedding-001")
            result = embedding_model.embed_content(
                content=text,
                task_type="retrieval_document"
            )
            
            # Convert the embedding to bytes
            import numpy as np
            embedding = np.array(result.embedding, dtype=np.float32)
            return embedding.tobytes()
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating embedding: {str(e)}")
            # Fall back to OpenAI embedding as a backup
            try:
                import openai
                client = openai.OpenAI(api_key=self.api_key)
                response = client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=text
                )
                
                embedding = np.array(response.data[0].embedding, dtype=np.float32)
                return embedding.tobytes()
            except Exception as fallback_error:
                if self.logger:
                    self.logger.error(f"Fallback embedding error: {str(fallback_error)}")
                raise Exception(f"Error creating embedding: {str(e)}, Fallback error: {str(fallback_error)}")
    
    def format_tool_calls(self, tool_calls: Any) -> List[Dict[str, Any]]:
        """
        Format Google Gemini tool calls to standardized format.
        
        Args:
            tool_calls: Google Gemini tool/function calls object
            
        Returns:
            List of standardized tool call objects
        """
        if not tool_calls:
            return []
        
        formatted_calls = []
        for i, tool_call in enumerate(tool_calls):
            # Handle Google's function_call format
            if hasattr(tool_call, 'function_call'):
                # Extract function name and arguments
                function_name = tool_call.function_call.name
                # Arguments in Google's format might be a string
                try:
                    if isinstance(tool_call.function_call.args, str):
                        arguments = tool_call.function_call.args
                    else:
                        arguments = json.dumps(tool_call.function_call.args)
                except Exception:
                    arguments = "{}"
                
                formatted_call = {
                    "id": f"call_{i}",  # Google doesn't provide IDs, so we create them
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": arguments
                    }
                }
            elif isinstance(tool_call, dict) and 'function_call' in tool_call:
                # Handle dictionary-like object with function_call key
                function_name = tool_call['function_call'].get('name')
                try:
                    if isinstance(tool_call['function_call'].get('args'), str):
                        arguments = tool_call['function_call'].get('args')
                    else:
                        arguments = json.dumps(tool_call['function_call'].get('args', {}))
                except Exception:
                    arguments = "{}"
                
                formatted_call = {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": arguments
                    }
                }
            else:
                # Skip unrecognized formats
                continue
            
            formatted_calls.append(formatted_call)
        
        return formatted_calls
    
    def format_tool_results(self, tool_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format tool results for Google Gemini.
        
        Args:
            tool_results: List of standardized tool result objects
            
        Returns:
            List of Google Gemini-compatible tool result objects
        """
        formatted_results = []
        for result in tool_results:
            formatted_result = {
                "role": "user",
                "parts": [{
                    "function_response": {
                        "name": result.get("name"),
                        "response": {
                            "content": result.get("content")
                        }
                    }
                }]
            }
            formatted_results.append(formatted_result)
        
        return formatted_results