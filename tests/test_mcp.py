"""
Unit tests for MCP functionality.
"""
import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import tempfile
import yaml

from yamllm.core.parser import YamlLMConfig, MCPConnectorSettings
from yamllm.mcp.client import MCPClient
from yamllm.mcp.connector import MCPConnector
from yamllm.providers.base import ToolDefinition


class TestMCPConnector(unittest.TestCase):
    """Test cases for MCP connector."""
    
    def setUp(self):
        """Set up test cases."""
        self.connector = MCPConnector(
            name="test_connector",
            url="https://example.com/mcp",
            authentication="test_token",
            description="Test MCP connector",
            tool_prefix="test"
        )
    
    def test_init(self):
        """Test initialization of MCP connector."""
        self.assertEqual(self.connector.name, "test_connector")
        self.assertEqual(self.connector.url, "https://example.com/mcp")
        self.assertEqual(self.connector.auth_token, "test_token")
        self.assertEqual(self.connector.description, "Test MCP connector")
        self.assertEqual(self.connector.tool_prefix, "test")
    
    @patch('requests.get')
    def test_discover_tools(self, mock_get):
        """Test discovering tools from MCP server."""
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "name": "tool1",
                "description": "Test tool 1",
                "parameters": {"type": "object", "properties": {}}
            }
        ]
        mock_get.return_value = mock_response
        
        # Call the method
        tools = self.connector.discover_tools()
        
        # Assertions
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["name"], "test_tool1")
        
        # Verify the request
        mock_get.assert_called_once_with(
            "https://example.com/mcp/tools",
            headers=self.connector._get_headers(),
            timeout=10
        )
    
    @patch('requests.post')
    def test_execute_tool(self, mock_post):
        """Test executing a tool on MCP server."""
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "test_result"}
        mock_post.return_value = mock_response
        
        # Call the method
        result = self.connector.execute_tool("test_tool1", {"param": "value"})
        
        # Assertions
        self.assertEqual(result, {"result": "test_result"})
        
        # Verify the request
        mock_post.assert_called_once_with(
            "https://example.com/mcp/tools/tool1/execute",
            headers=self.connector._get_headers(),
            json={"param": "value"},
            timeout=30
        )


class TestMCPClient(unittest.TestCase):
    """Test cases for MCP client."""
    
    def setUp(self):
        """Set up test cases."""
        self.client = MCPClient()
        
        # Create a mock connector
        self.connector = MagicMock()
        self.connector.name = "test_connector"
        self.connector.discover_tools.return_value = [
            {
                "name": "test_tool1",
                "description": "Test tool 1",
                "parameters": {"type": "object", "properties": {}}
            }
        ]
        
        # Register the connector
        self.client.register_connector(self.connector)
    
    def test_register_connector(self):
        """Test registering a connector."""
        self.assertIn("test_connector", self.client.connectors)
        self.assertEqual(self.client.connectors["test_connector"], self.connector)
    
    def test_discover_all_tools(self):
        """Test discovering tools from all connectors."""
        tools = self.client.discover_all_tools()
        self.assertIn("test_connector", tools)
        self.assertEqual(len(tools["test_connector"]), 1)
        self.assertEqual(tools["test_connector"][0]["name"], "test_tool1")
    
    def test_execute_tool(self):
        """Test executing a tool."""
        self.connector.execute_tool.return_value = {"result": "test_result"}
        
        result = self.client.execute_tool("test_connector", "test_tool1", {"param": "value"})
        
        self.assertEqual(result, {"result": "test_result"})
        self.connector.execute_tool.assert_called_once_with("test_tool1", {"param": "value"})
    
    def test_convert_mcp_tools_to_definitions(self):
        """Test converting MCP tools to tool definitions."""
        definitions = self.client.convert_mcp_tools_to_definitions()
        
        self.assertEqual(len(definitions), 1)
        self.assertIsInstance(definitions[0], ToolDefinition)
        self.assertEqual(definitions[0].name, "test_tool1")
        self.assertEqual(definitions[0].description, "Test tool 1")
        self.assertEqual(definitions[0].mcp_connector_name, "test_connector")
        self.assertEqual(definitions[0].mcp_tool_id, "test_tool1")


if __name__ == '__main__':
    unittest.main()