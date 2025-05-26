"""
MCP Client implementation for YAMLLM.

This module provides the client implementation for interacting with MCP servers.
"""

import logging
from typing import Dict, List, Any, Optional
from .connector import MCPConnector
from yamllm.providers.base import ToolDefinition


class MCPClient:
    """
    Client for Model Context Protocol (MCP) servers.
    
    This class provides methods for:
    - Managing MCP connectors
    - Discovering tools from MCP servers
    - Executing MCP tools
    - Converting MCP tool schemas to YAMLLM's tool definitions
    """
    
    def __init__(self):
        """Initialize the MCP client."""
        self.connectors = {}
        self.logger = logging.getLogger(__name__)
    
    def register_connector(self, connector: MCPConnector) -> None:
        """
        Register an MCP connector.
        
        Args:
            connector (MCPConnector): The connector to register.
        """
        self.connectors[connector.name] = connector
        self.logger.debug(f"Registered MCP connector: {connector.name}")
    
    def discover_all_tools(self, force_refresh: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """
        Discover tools from all registered MCP connectors.
        
        Args:
            force_refresh (bool): Whether to force a refresh of cached tools.
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Dictionary mapping connector names to their tool definitions.
        """
        all_tools = {}
        
        for name, connector in self.connectors.items():
            try:
                tools = connector.discover_tools(force_refresh=force_refresh)
                all_tools[name] = tools
            except Exception as e:
                self.logger.error(f"Error discovering tools from connector {name}: {str(e)}")
                # Continue with other connectors even if one fails
        
        return all_tools
    
    def execute_tool(self, connector_name: str, tool_id: str, parameters: Dict[str, Any]) -> Any:
        """
        Execute a tool on an MCP server.
        
        Args:
            connector_name (str): The name of the connector to use.
            tool_id (str): The ID of the tool to execute.
            parameters (Dict[str, Any]): The parameters for the tool.
            
        Returns:
            Any: The result of the tool execution.
            
        Raises:
            ValueError: If the connector is not found.
        """
        if connector_name not in self.connectors:
            raise ValueError(f"MCP connector '{connector_name}' not found")
        
        connector = self.connectors[connector_name]
        return connector.execute_tool(tool_id, parameters)
    
    def convert_mcp_tools_to_definitions(self) -> List[ToolDefinition]:
        """
        Convert MCP tools to YAMLLM tool definitions.
        
        Returns:
            List[ToolDefinition]: List of tool definitions.
        """
        tool_definitions = []
        all_tools = self.discover_all_tools()
        
        for connector_name, tools in all_tools.items():
            for tool in tools:
                # Extract tool information
                name = tool.get("name", "unknown_tool")
                description = tool.get("description", f"Tool from MCP connector {connector_name}")
                parameters = tool.get("parameters", {"type": "object", "properties": {}})
                
                # Create a tool definition
                tool_definition = ToolDefinition(
                    name=name,
                    description=description,
                    parameters=parameters
                )
                
                # Store connector name for later use when executing the tool
                tool_definition.mcp_connector_name = connector_name
                tool_definition.mcp_tool_id = tool.get("id", name)
                
                tool_definitions.append(tool_definition)
        
        return tool_definitions