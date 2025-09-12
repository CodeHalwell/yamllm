"""
Base provider interface for YAMLLM.

This module defines the base interface that all LLM providers must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class Message:
    """
    Standardized message format for communication with LLM providers.
    
    This class provides a consistent interface for messages across different providers.
    """
    
    def __init__(self, role: str, content: str, name: Optional[str] = None):
        """
        Initialize a message.
        
        Args:
            role (str): The role of the message sender (e.g., "system", "user", "assistant").
            content (str): The content of the message.
            name (str, optional): The name of the message sender.
        """
        self.role = role
        self.content = content
        self.name = name
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the message to a dictionary format.
        
        Returns:
            Dict[str, Any]: The message as a dictionary.
        """
        message_dict = {
            "role": self.role,
            "content": self.content
        }
        if self.name:
            message_dict["name"] = self.name
        return message_dict


class ToolCall:
    """
    Standardized tool call format for LLM providers.
    
    This class provides a consistent interface for tool calls across different providers.
    """
    
    def __init__(self, tool_id: str, name: str, arguments: Dict[str, Any]):
        """
        Initialize a tool call.
        
        Args:
            tool_id (str): The ID of the tool.
            name (str): The name of the tool.
            arguments (Dict[str, Any]): The arguments for the tool call.
        """
        self.tool_id = tool_id
        self.name = name
        self.arguments = arguments


class ToolDefinition:
    """
    Standardized tool definition format for LLM providers.
    
    This class provides a consistent interface for tool definitions across different providers.
    """
    
    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        """
        Initialize a tool definition.
        
        Args:
            name (str): The name of the tool.
            description (str): The description of the tool.
            parameters (Dict[str, Any]): The parameters schema for the tool.
        """
        self.name = name
        self.description = description
        self.parameters = parameters
        
        # MCP-specific fields
        self.is_mcp_tool = False
        self.mcp_connector_name = None
        self.mcp_tool_id = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the tool definition to a dictionary format.
        
        Returns:
            Dict[str, Any]: The tool definition as a dictionary.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


class BaseProvider(ABC):
    """
    Base provider interface for YAMLLM.
    
    This abstract class defines the interface that all LLM providers must implement.
    """
    
    @abstractmethod
    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        """
        Initialize the provider.
        
        Args:
            api_key (str): The API key for the provider.
            base_url (str, optional): The base URL for the provider's API.
            **kwargs: Additional provider-specific parameters.
        """
        pass
    
    @abstractmethod
    def get_completion(self, messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int, top_p: float, stop_sequences: Optional[List[str]] = None, tools: Optional[List[Dict[str, Any]]] = None, tool_choice: str = "auto") -> Any:
        """
        Get a completion from the model.
        
        Args:
            messages (List[Dict[str, str]]): The messages to send to the model.
            model (str): The model to use.
            temperature (float): The temperature parameter for the model.
            max_tokens (int): The maximum number of tokens to generate.
            top_p (float): The top_p parameter for the model.
            stop_sequences (List[str], optional): Sequences that will stop generation.
            tools (List[Dict[str, Any]], optional): Tool definitions.
            tool_choice (str, optional): The tool choice strategy.
            
        Returns:
            Any: The response from the model.
        """
        pass

    @abstractmethod
    def get_streaming_completion(self, messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int, top_p: float, stop_sequences: Optional[List[str]] = None, tools: Optional[List[Dict[str, Any]]] = None, tool_choice: str = "auto") -> Any:
        """
        Get a streaming completion from the model.
        
        Args:
            messages (List[Dict[str, str]]): The messages to send to the model.
            model (str): The model to use.
            temperature (float): The temperature parameter for the model.
            max_tokens (int): The maximum number of tokens to generate.
            top_p (float): The top_p parameter for the model.
            stop_sequences (List[str], optional): Sequences that will stop generation.
            tools (List[Dict[str, Any]], optional): Tool definitions.
            tool_choice (str, optional): The tool choice strategy.
            
        Returns:
            Any: The streaming response from the model.
        """
        pass

    def format_tool_calls(self, tool_calls: Any) -> List[Dict[str, Any]]:
        """
        Format tool calls from the provider into a standardized format.
        
        Args:
            tool_calls (Any): The tool calls from the provider.
            
        Returns:
            List[Dict[str, Any]]: The formatted tool calls.
        """
        return tool_calls

    def close(self):
        """
        Close any open connections.
        """
        pass
