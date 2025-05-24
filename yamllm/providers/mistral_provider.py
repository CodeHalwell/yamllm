"""
MistralAI provider implementation for YAMLLM.

This module provides an implementation of the BaseProvider interface for MistralAI.
"""

from typing import Dict, List, Any, Optional
from rich.console import Console
from rich.markdown import Markdown

from yamllm.providers.base import Message, ToolDefinition
from yamllm.providers.openai_provider import OpenAIProvider


class MistralProvider(OpenAIProvider):
    """
    MistralAI provider implementation.
    
    This class implements the BaseProvider interface for MistralAI.
    Since MistralAI uses an OpenAI-compatible API with some differences,
    it inherits from OpenAIProvider and overrides specific methods.
    """
    
    # Real-time query keywords - similar to GoogleGemini implementation
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
        Initialize the MistralAI provider.
        
        Args:
            api_key (str): The API key for MistralAI.
            model (str): The model to use.
            base_url (str, optional): The base URL for the MistralAI API.
            **kwargs: Additional provider-specific parameters.
        """
        super().__init__(api_key, model, base_url, **kwargs)
    
    def prepare_completion_params(self, messages: List[Message], temperature: float, max_tokens: int, 
                                 top_p: float, stop_sequences: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Override to prepare parameters compatible with Mistral's API requirements.
        
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
    
    def handle_streaming_with_tool_detection(self, messages: List[Message], params: Dict[str, Any], 
                                           tools: Optional[List[ToolDefinition]] = None) -> str:
        """
        Override to handle streaming with tool detection for Mistral's API.
        
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
    
    def handle_non_streaming_response(self, messages: List[Message], params: Dict[str, Any], 
                                     tools: Optional[List[ToolDefinition]] = None) -> str:
        """
        Override to handle non-streaming response for Mistral's API.
        
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