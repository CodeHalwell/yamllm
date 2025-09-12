"""
Tool orchestration component for YAMLLM.

This module handles tool registration, execution, and coordination,
extracted from the main LLM class for better separation of concerns.
"""

from typing import Dict, List, Any, Optional
import os
import logging
import asyncio

from yamllm.tools.manager import ToolManager
from yamllm.tools.security import SecurityManager
from yamllm.tools.utility_tools import (
    WebSearch, Calculator, TimezoneTool, UnitConverter, WeatherTool,
    WebScraper, DateTimeTool, UUIDTool, RandomStringTool, RandomNumberTool,
    Base64EncodeTool, Base64DecodeTool, HashTool, JSONTool, RegexExtractTool,
    LoremIpsumTool, FileReadTool, FileSearchTool, CSVPreviewTool,
    URLMetadataTool, WebHeadlinesTool, ToolsHelpTool
)


class ToolOrchestrator:
    """
    Manages tool registration, execution, and coordination for LLM interactions.
    
    This class encapsulates all tool-related functionality that was
    previously embedded in the main LLM class.
    """
    
    # Tool registry mapping names to classes
    TOOL_REGISTRY = {
        "web_search": WebSearch,
        "calculator": Calculator,
        "timezone": TimezoneTool,
        "unit_converter": UnitConverter,
        "weather": WeatherTool,
        "web_scraper": WebScraper,
        "datetime": DateTimeTool,
        "uuid": UUIDTool,
        "random_string": RandomStringTool,
        "random_number": RandomNumberTool,
        "base64_encode": Base64EncodeTool,
        "base64_decode": Base64DecodeTool,
        "hash_text": HashTool,
        "json_tool": JSONTool,
        "regex_extract": RegexExtractTool,
        "lorem_ipsum": LoremIpsumTool,
        "file_read": FileReadTool,
        "file_search": FileSearchTool,
        "csv_preview": CSVPreviewTool,
        "url_metadata": URLMetadataTool,
        "web_headlines": WebHeadlinesTool,
    }
    
    # Tool pack definitions
    TOOL_PACKS = {
        "common": [
            "calculator", "datetime", "uuid", "random_string",
            "json_tool", "regex_extract", "lorem_ipsum"
        ],
        "web": [
            "web_search", "web_scraper", "url_metadata",
            "web_headlines", "weather"
        ],
        "files": ["file_read", "file_search", "csv_preview"],
        "crypto": ["hash_text", "base64_encode", "base64_decode"],
        "numbers": ["random_number", "unit_converter"],
        "time": ["datetime", "timezone"],
        "dev": [
            "json_tool", "regex_extract", "hash_text", "base64_encode",
            "base64_decode", "file_read", "file_search", "csv_preview",
            "uuid", "random_string"
        ],
        "all": []  # Special case - includes all tools
    }
    
    def __init__(
        self,
        enabled: bool = True,
        tool_list: List[str] = None,
        tool_packs: List[str] = None,
        tool_timeout: int = 30,
        security_config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
        mcp_client: Optional[Any] = None
    ):
        """
        Initialize the tool orchestrator.
        
        Args:
            enabled: Whether tools are enabled
            tool_list: List of individual tools to enable
            tool_packs: List of tool packs to enable
            tool_timeout: Timeout for tool execution in seconds
            security_config: Security configuration for tools
            logger: Logger instance
            mcp_client: MCP client for external tools
        """
        self.enabled = enabled
        self.tool_list = tool_list or []
        self.tool_packs = tool_packs or []
        self.tool_timeout = tool_timeout
        self.logger = logger or logging.getLogger(__name__)
        self.mcp_client = mcp_client
        
        # Initialize security manager
        security_config = security_config or {}
        self.security_manager = SecurityManager(
            allowed_paths=security_config.get('allowed_paths', []),
            safe_mode=security_config.get('safe_mode', False),
            allow_network=security_config.get('allow_network', True),
            allow_filesystem=security_config.get('allow_filesystem', True),
            blocked_domains=security_config.get('blocked_domains', [])
        )
        
        # Initialize tool manager
        self.tool_manager = ToolManager(timeout=self.tool_timeout, logger=self.logger)
        
        # Track execution for circular dependency detection
        self._execution_stack = []
        
        if self.enabled:
            self._register_tools()
    
    def _register_tools(self):
        """Register configured tools with the tool manager."""
        try:
            # Expand tool selections
            selected_tools = self._expand_tool_selection()
            
            self.logger.debug(f"Selected tools to register: {selected_tools}")
            
            # Register each tool
            for tool_name in selected_tools:
                self._register_tool(tool_name)
            
            # Always register help tool last
            self.tool_manager.register(ToolsHelpTool(self.tool_manager))
            
        except Exception as e:
            self.logger.error(f"Tool registration error: {e}")
    
    def _expand_tool_selection(self) -> List[str]:
        """Expand tool packs and deduplicate tool selection."""
        selected = []
        
        # Add tools from packs
        for pack in self.tool_packs:
            if pack == "all":
                selected.extend(self.TOOL_REGISTRY.keys())
            else:
                selected.extend(self.TOOL_PACKS.get(pack, []))
        
        # Add individual tools
        selected.extend(self.tool_list)
        
        # Deduplicate while preserving order
        seen = set()
        return [t for t in selected if not (t in seen or seen.add(t))]
    
    def _register_tool(self, tool_name: str):
        """Register a single tool."""
        tool_class = self.TOOL_REGISTRY.get(tool_name)
        if not tool_class:
            self.logger.warning(f"Tool '{tool_name}' not found in registry")
            return
        
        try:
            # Create tool instance with appropriate configuration
            if tool_name == "weather":
                tool = tool_class(
                    api_key=os.environ.get("WEATHER_API_KEY"),
                    security_manager=self.security_manager
                )
            elif tool_name in {"web_search", "web_scraper"}:
                tool = tool_class(security_manager=self.security_manager)
            elif tool_name in {"url_metadata", "web_headlines"}:
                tool = tool_class()  # These inherit security through base class
            elif tool_name in {"file_read", "file_search", "csv_preview"}:
                tool = tool_class(security_manager=self.security_manager)
            else:
                tool = tool_class()
            
            self.tool_manager.register(tool)
            self.logger.debug(f"Successfully registered tool: {tool_name}")
            
        except Exception as e:
            self.logger.warning(f"Failed to register tool '{tool_name}': {e}")
    
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Get all tool definitions for API calls.
        
        Returns:
            List of tool definitions in provider-expected format
        """
        if not self.enabled:
            return []
        
        definitions = []
        
        # Add local tool definitions
        try:
            definitions.extend(self.tool_manager.get_tool_definitions())
        except Exception as e:
            self.logger.debug(f"Unable to get local tool definitions: {e}")
        
        # Add MCP tool definitions if available
        if self.mcp_client:
            try:
                mcp_tools = asyncio.run(self.mcp_client.convert_mcp_tools_to_definitions())
                definitions.extend([tool.to_dict() for tool in mcp_tools])
                self.logger.debug(f"Added {len(mcp_tools)} tools from MCP connectors")
            except Exception as e:
                self.logger.error(f"Error adding MCP tools: {e}")
        
        return definitions
    
    def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """
        Execute a tool with circular dependency detection.
        
        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            
        Returns:
            Tool execution result
            
        Raises:
            Exception: If circular dependency detected or execution fails
        """
        # Check for circular dependencies
        if tool_name in self._execution_stack:
            raise Exception(f"Circular tool dependency detected: {tool_name}")
        
        self._execution_stack.append(tool_name)
        try:
            # Check if this is an MCP tool
            if self.mcp_client:
                try:
                    mcp_tools = asyncio.run(self.mcp_client.convert_mcp_tools_to_definitions())
                    for tool in mcp_tools:
                        if tool.name == tool_name:
                            return asyncio.run(self.mcp_client.execute_tool(
                                connector_name=tool.mcp_connector_name,
                                tool_id=tool.mcp_tool_id,
                                parameters=tool_args
                            ))
                except Exception as e:
                    self.logger.error(f"Error checking MCP tools: {e}")
            
            # Execute via tool manager
            return self.tool_manager.execute(tool_name, tool_args)
            
        finally:
            self._execution_stack.pop()
    
    def reset_execution_stack(self):
        """Reset the execution stack (call between independent requests)."""
        self._execution_stack.clear()
    
    def close(self):
        """Clean up resources."""
        # Tool manager doesn't currently have cleanup, but might in future
        if hasattr(self.tool_manager, 'close'):
            self.tool_manager.close()