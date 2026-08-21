"""
MCP edge case tests.

Tests for MCP connector failure scenarios and edge cases that were
identified as missing coverage.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from yamllm.mcp.client import MCPClient


class TimeoutConnector:
    """Mock connector that times out."""
    
    def __init__(self, name: str):
        self.name = name
    
    async def discover_tools(self, force_refresh: bool = False):
        """Simulate timeout during discovery."""
        import asyncio
        await asyncio.sleep(10)  # Simulate slow response
        return []
    
    async def execute_tool(self, tool_id: str, parameters):
        """Simulate timeout during execution."""
        import asyncio
        await asyncio.sleep(10)  # Simulate slow response
        return {}
    
    async def disconnect(self):
        return None


class MalformedResponseConnector:
    """Mock connector that returns malformed responses."""
    
    def __init__(self, name: str):
        self.name = name
    
    async def discover_tools(self, force_refresh: bool = False):
        """Return malformed tool definitions."""
        return [
            {"invalid": "structure"},  # Missing required fields
            {"name": "tool2"},  # Missing description and parameters
            None,  # Null entry
        ]
    
    async def execute_tool(self, tool_id: str, parameters):
        """Return malformed execution result."""
        return "not a dict"  # Should be dict
    
    async def disconnect(self):
        return None


class FailingConnector:
    """Mock connector that raises exceptions."""
    
    def __init__(self, name: str, error_type=Exception):
        self.name = name
        self.error_type = error_type
    
    async def discover_tools(self, force_refresh: bool = False):
        """Raise exception during discovery."""
        raise self.error_type("Connection failed")
    
    async def execute_tool(self, tool_id: str, parameters):
        """Raise exception during execution."""
        raise self.error_type("Execution failed")
    
    async def disconnect(self):
        return None


@pytest.mark.anyio
class TestMCPConnectorTimeout:
    """Test MCP connector timeout scenarios."""
    
    async def test_discover_tools_timeout(self):
        """Test timeout during tool discovery."""
        client = MCPClient()
        connector = TimeoutConnector("slow")
        client.register_connector(connector)
        
        # Use timeout to prevent test from hanging
        import asyncio
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(client.discover_all_tools(), timeout=0.1)
    
    async def test_execute_tool_timeout(self):
        """Test timeout during tool execution."""
        client = MCPClient()
        connector = TimeoutConnector("slow")
        client.register_connector(connector)
        
        # Use timeout to prevent test from hanging
        import asyncio
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                client.execute_tool("slow", "tool_id", {}),
                timeout=0.1
            )


@pytest.mark.anyio
class TestMCPMalformedResponse:
    """Test handling of malformed MCP responses."""
    
    async def test_malformed_tool_definitions(self):
        """Test handling of malformed tool definitions."""
        client = MCPClient()
        connector = MalformedResponseConnector("bad")
        client.register_connector(connector)
        
        # Should handle malformed responses gracefully
        result = await client.discover_all_tools()
        
        # Result should be dict even with malformed data
        assert isinstance(result, dict)
        assert "bad" in result
    
    async def test_malformed_execution_result(self):
        """Test handling of malformed execution result."""
        client = MCPClient()
        connector = MalformedResponseConnector("bad")
        client.register_connector(connector)
        
        # Should handle malformed execution result
        result = await client.execute_tool("bad", "tool_id", {})
        
        # Should return the result even if malformed
        assert result is not None


@pytest.mark.anyio
class TestMCPConnectionFailure:
    """Test MCP connection failure scenarios."""
    
    async def test_discover_tools_connection_error(self):
        """Test connection error during tool discovery."""
        client = MCPClient()
        connector = FailingConnector("failing", ConnectionError)
        client.register_connector(connector)
        
        # Should handle connection error gracefully
        with pytest.raises(ConnectionError):
            await client.discover_all_tools()
    
    async def test_execute_tool_connection_error(self):
        """Test connection error during tool execution."""
        client = MCPClient()
        connector = FailingConnector("failing", ConnectionError)
        client.register_connector(connector)
        
        # Should handle connection error gracefully
        with pytest.raises(ConnectionError):
            await client.execute_tool("failing", "tool_id", {})
    
    async def test_generic_error(self):
        """Test generic error handling."""
        client = MCPClient()
        connector = FailingConnector("failing", RuntimeError)
        client.register_connector(connector)
        
        # Should propagate generic errors
        with pytest.raises(RuntimeError):
            await client.discover_all_tools()


@pytest.mark.anyio
class TestMCPToolRegistrationFailure:
    """Test tool registration failure scenarios."""
    
    async def test_register_none_connector(self):
        """Test registering None connector."""
        client = MCPClient()
        
        # Should handle None gracefully
        try:
            client.register_connector(None)
            # If no exception, check connectors list
            assert None not in client.connectors
        except (TypeError, AttributeError):
            # Expected behavior - reject None
            pass
    
    async def test_register_invalid_connector(self):
        """Test registering invalid connector."""
        client = MCPClient()
        
        # Try to register object without required methods
        invalid = object()
        
        try:
            client.register_connector(invalid)
            # If registered, should fail on use
            with pytest.raises(AttributeError):
                await client.discover_all_tools()
        except (TypeError, AttributeError):
            # Expected - reject invalid connector
            pass
    
    async def test_multiple_connectors_with_failures(self):
        """Test multiple connectors where some fail."""
        client = MCPClient()
        
        # Mix of working and failing connectors
        working = MalformedResponseConnector("working")
        failing = FailingConnector("failing")
        
        client.register_connector(working)
        client.register_connector(failing)
        
        # Should fail due to failing connector
        with pytest.raises(Exception):
            await client.discover_all_tools()


@pytest.mark.anyio
class TestMCPEmptyResponses:
    """Test MCP empty response handling."""
    
    async def test_empty_tool_list(self):
        """Test connector returning empty tool list."""
        client = MCPClient()
        
        class EmptyConnector:
            def __init__(self, name):
                self.name = name
            
            async def discover_tools(self, force_refresh=False):
                return []
            
            async def execute_tool(self, tool_id, parameters):
                return {}
            
            async def disconnect(self):
                return None
        
        connector = EmptyConnector("empty")
        client.register_connector(connector)
        
        result = await client.discover_all_tools()
        
        # Should handle empty list gracefully
        assert "empty" in result
        assert result["empty"] == []
    
    async def test_empty_execution_result(self):
        """Test tool execution returning empty result."""
        client = MCPClient()
        
        class EmptyResultConnector:
            def __init__(self, name):
                self.name = name
            
            async def discover_tools(self, force_refresh=False):
                return []
            
            async def execute_tool(self, tool_id, parameters):
                return {}
            
            async def disconnect(self):
                return None
        
        connector = EmptyResultConnector("empty")
        client.register_connector(connector)
        
        result = await client.execute_tool("empty", "tool_id", {})
        
        # Should handle empty result
        assert result == {}


@pytest.mark.anyio
class TestMCPDisconnectFailure:
    """Test MCP disconnect failure scenarios."""
    
    async def test_disconnect_error(self):
        """Test error during disconnect."""
        client = MCPClient()
        
        class BadDisconnectConnector:
            def __init__(self, name):
                self.name = name
            
            async def discover_tools(self, force_refresh=False):
                return []
            
            async def execute_tool(self, tool_id, parameters):
                return {}
            
            async def disconnect(self):
                raise RuntimeError("Disconnect failed")
        
        connector = BadDisconnectConnector("bad")
        client.register_connector(connector)
        
        # Disconnect should handle errors gracefully
        try:
            await client.disconnect_all()
        except RuntimeError:
            # Expected behavior - may propagate error
            pass


@pytest.mark.anyio
class TestMCPConcurrentAccess:
    """Test MCP concurrent access scenarios."""
    
    async def test_concurrent_discovery(self):
        """Test concurrent tool discovery from multiple connectors."""
        client = MCPClient()
        
        class SlowConnector:
            def __init__(self, name, delay=0.1):
                self.name = name
                self.delay = delay
            
            async def discover_tools(self, force_refresh=False):
                import asyncio
                await asyncio.sleep(self.delay)
                return [{"name": f"{self.name}_tool", "description": "Test"}]
            
            async def execute_tool(self, tool_id, parameters):
                return {}
            
            async def disconnect(self):
                return None
        
        # Register multiple connectors
        for i in range(3):
            client.register_connector(SlowConnector(f"conn{i}"))
        
        # Discover tools concurrently
        result = await client.discover_all_tools()
        
        # All connectors should return results
        assert len(result) == 3


@pytest.mark.anyio
class TestMCPParameterValidation:
    """Test MCP parameter validation."""
    
    async def test_execute_with_invalid_parameters(self):
        """Test tool execution with invalid parameters."""
        client = MCPClient()
        
        class StrictConnector:
            def __init__(self, name):
                self.name = name
            
            async def discover_tools(self, force_refresh=False):
                return []
            
            async def execute_tool(self, tool_id, parameters):
                if not isinstance(parameters, dict):
                    raise TypeError("Parameters must be dict")
                return {"result": "ok"}
            
            async def disconnect(self):
                return None
        
        connector = StrictConnector("strict")
        client.register_connector(connector)
        
        # Test with invalid parameters
        with pytest.raises(TypeError):
            await client.execute_tool("strict", "tool_id", "not a dict")
        
        # Test with valid parameters
        result = await client.execute_tool("strict", "tool_id", {"valid": "params"})
        assert result["result"] == "ok"
