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
    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None, **kwargs):
        """
        Initialize the provider.
        
        Args:
            api_key (str): The API key for the provider.
            model (str): The model to use.
            base_url (str, optional): The base URL for the provider's API.
            **kwargs: Additional provider-specific parameters.
        """
        pass
    
    @abstractmethod
    def prepare_completion_params(self, messages: List[Message], temperature: float, max_tokens: int, 
                                 top_p: float, stop_sequences: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Prepare completion parameters for the provider's API.
        
        Args:
            messages (List[Message]): The messages to send to the model.
            temperature (float): The temperature parameter for the model.
            max_tokens (int): The maximum number of tokens to generate.
            top_p (float): The top_p parameter for the model.
            stop_sequences (List[str], optional): Sequences that will stop generation.
            
        Returns:
            Dict[str, Any]: The parameters for the API request.
        """
        pass
    
    @abstractmethod
    def handle_streaming_response(self, messages: List[Message], params: Dict[str, Any]) -> str:
        """
        Handle streaming response from the model.
        
        Args:
            messages (List[Message]): The messages sent to the model.
            params (Dict[str, Any]): The parameters for the API request.
            
        Returns:
            str: The concatenated response text.
        """
        pass
    
    @abstractmethod
    def handle_non_streaming_response(self, messages: List[Message], params: Dict[str, Any], 
                                     tools: Optional[List[ToolDefinition]] = None) -> str:
        """
        Handle non-streaming response from the model.
        
        Args:
            messages (List[Message]): The messages sent to the model.
            params (Dict[str, Any]): The parameters for the API request.
            tools (List[ToolDefinition], optional): Tool definitions.
            
        Returns:
            str: The response text.
        """
        pass
    
    @abstractmethod
    def handle_streaming_with_tool_detection(self, messages: List[Message], params: Dict[str, Any], 
                                           tools: Optional[List[ToolDefinition]] = None) -> str:
        """
        Handle streaming with tool detection.
        
        Args:
            messages (List[Message]): The messages sent to the model.
            params (Dict[str, Any]): The parameters for the API request.
            tools (List[ToolDefinition], optional): Tool definitions.
            
        Returns:
            str: The response text.
        """
        pass
    
    @abstractmethod
    def process_tool_calls(self, messages: List[Message], model_message: Any, 
                          execute_tool_func: callable, max_iterations: int = 5) -> str:
        """
        Process tool calls from the model.
        
        Args:
            messages (List[Message]): The messages sent to the model.
            model_message (Any): The message from the model containing tool calls.
            execute_tool_func (callable): Function to execute a tool.
            max_iterations (int, optional): Maximum number of tool call iterations.
            
        Returns:
            str: The final response text.
        """
        pass
    
    @abstractmethod
    def create_embedding(self, text: str) -> bytes:
        """
        Create an embedding for the given text.
        
        Args:
            text (str): The text to create an embedding for.
            
        Returns:
            bytes: The embedding as bytes.
        """
        pass