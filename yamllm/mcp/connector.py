"""
MCP Connector implementation for YAMLLM.
"""

import asyncio
import json
import os
import re
import logging
import httpx
import websockets
from enum import Enum
from typing import Any, Dict, List, Optional
from websockets.exceptions import ConnectionClosed

from yamllm.core.exceptions import YAMLLMException


class MCPError(YAMLLMException):
    """MCP-related errors."""
    pass


class MCPTransportType(Enum):
    WEBSOCKET = "websocket"
    HTTP = "http"
    STDIO = "stdio"


class MCPConnector:
    """Connector for MCP servers."""
    
    def __init__(self, name, url, authentication=None, description=None, tool_prefix=None, transport="http"):
        self.name = name
        self.url = url.rstrip("/") if transport == "http" else url
        self.description = description or f"MCP connector for {name}"
        self.tool_prefix = tool_prefix
        self.transport = MCPTransportType(transport) if isinstance(transport, str) else transport
        self.auth_token = self._process_auth(authentication)
        self.logger = logging.getLogger(__name__)
        self._cached_tools = None
        # Map of displayed tool id -> remote tool id (without prefix)
        self._id_map: Dict[str, str] = {}
        self._connection = None
        self._process = None
        self._connected = False
        self._http_client = httpx.AsyncClient(headers=self._get_headers(), timeout=10)

    def _process_auth(self, auth_str):
        """Process authentication string, resolving environment variables."""
        if not auth_str:
            return None
            
        env_var_match = re.match(r"\$\{(.+)\}", auth_str)
        if env_var_match:
            env_var_name = env_var_match.group(1)
            return os.environ.get(env_var_name)
        return auth_str

    # HTTP helpers ---------------------------------------------------------
    def _get_headers(self) -> Dict[str, str]:
        """Build standard headers for MCP HTTP endpoints."""
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "yamllm-mcp/1.0",
        }
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    # Connection Management -----------------------------------------------
    async def connect(self) -> None:
        """Establish connection based on transport type."""
        try:
            if self.transport == MCPTransportType.WEBSOCKET:
                await self._connect_websocket()
            elif self.transport == MCPTransportType.HTTP:
                await self._connect_http()
            elif self.transport == MCPTransportType.STDIO:
                await self._connect_stdio()
            else:
                raise MCPError(f"Unsupported transport: {self.transport}")
            
            self._connected = True
            self.logger.info(f"Connected to MCP server {self.name} via {self.transport.value}")
        except Exception as e:
            raise MCPError(f"Failed to connect to {self.name}: {e}") from e
    
    async def _connect_websocket(self) -> None:
        """Connect via WebSocket."""
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        self._connection = await websockets.connect(
            self.url,
            extra_headers=headers,
            ping_interval=30,
            ping_timeout=10
        )
    
    async def _connect_http(self) -> None:
        """Connect via HTTP (no persistent connection needed)."""
        # Test connection
        try:
            response = await self._http_client.get(f"{self.url}/health")
            if response.status_code >= 500:
                raise MCPError(f"Server error: {response.status_code}")
        except httpx.RequestError as e:
            self.logger.warning(f"Health check failed for {self.name}: {e}")
    
    async def _connect_stdio(self) -> None:
        """Connect via stdio subprocess."""
        cmd_parts = self.url.split()
        if not cmd_parts:
            raise MCPError("Empty stdio command")
        
        self._process = await asyncio.create_subprocess_exec(
            *cmd_parts,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
    
    async def disconnect(self) -> None:
        """Disconnect from MCP server."""
        try:
            if self.transport == MCPTransportType.WEBSOCKET and self._connection:
                await self._connection.close()
            elif self.transport == MCPTransportType.STDIO and self._process:
                self._process.terminate()
                await self._process.wait()
            
            await self._http_client.aclose()
            self._connected = False
            self.logger.info(f"Disconnected from MCP server {self.name}")
        except Exception as e:
            self.logger.error(f"Error disconnecting from {self.name}: {e}")

    # Discovery ------------------------------------------------------------
    async def discover_tools(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Discover tools from the MCP server via a simple HTTP endpoint.

        Expects a JSON array of tool objects with keys: name, description, parameters.
        Applies the optional tool_prefix to the displayed name and preserves the
        original server tool id in the 'id' field.
        """
        if self._cached_tools is not None and not force_refresh:
            return self._cached_tools
        
        if not self._connected:
            await self.connect()
        
        try:
            if self.transport == MCPTransportType.HTTP:
                endpoint = f"{self.url}/tools"
                resp = await self._http_client.get(endpoint)
                data = resp.json() or []
            elif self.transport == MCPTransportType.WEBSOCKET:
                data = await self._send_websocket_message({
                    "jsonrpc": "2.0",
                    "id": "discover_tools",
                    "method": "tools/list",
                    "params": {}
                })
                data = data.get("result", {}).get("tools", []) if data else []
            elif self.transport == MCPTransportType.STDIO:
                data = await self._send_stdio_message({
                    "jsonrpc": "2.0",
                    "id": "discover_tools",
                    "method": "tools/list",
                    "params": {}
                })
                data = data.get("result", {}).get("tools", []) if data else []
            else:
                raise MCPError(f"Unsupported transport for tool discovery: {self.transport}")
        except Exception as e:
            self.logger.error(f"Tool discovery failed for {self.name}: {e}")
            return []

        tools: List[Dict[str, Any]] = []
        self._id_map.clear()
        for item in data:
            remote_name = item.get("name") or item.get("id") or "tool"
            display_name = f"{self.tool_prefix}_{remote_name}" if self.tool_prefix else remote_name
            # Track mapping for later execution routing
            self._id_map[display_name] = remote_name
            tools.append(
                {
                    "id": remote_name,
                    "name": display_name,
                    "description": item.get("description") or f"Tool {remote_name}",
                    "parameters": item.get("parameters") or {"type": "object", "properties": {}},
                }
            )

        self._cached_tools = tools
        return tools

    async def _send_websocket_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send message via WebSocket and return response."""
        if not self._connection:
            raise MCPError("WebSocket not connected")
        
        try:
            await self._connection.send(json.dumps(message))
            response = await asyncio.wait_for(self._connection.recv(), timeout=30)
            return json.loads(response)
        except (ConnectionClosed, asyncio.TimeoutError) as e:
            raise MCPError(f"WebSocket communication failed: {e}") from e
    
    async def _send_stdio_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send message via stdio and return response."""
        if not self._process:
            raise MCPError("Stdio process not connected")
        
        try:
            # Send message
            message_str = json.dumps(message) + "\n"
            self._process.stdin.write(message_str.encode())
            await self._process.stdin.drain()
            
            # Read response
            response_line = await asyncio.wait_for(
                self._process.stdout.readline(), 
                timeout=30
            )
            return json.loads(response_line.decode())
        except (asyncio.TimeoutError, json.JSONDecodeError) as e:
            raise MCPError(f"Stdio communication failed: {e}") from e

    # Execution ------------------------------------------------------------
    async def execute_tool(self, tool_id: str, parameters: Dict[str, Any]) -> Any:
        """Execute a specific tool by id (display name) with parameters.

        Resolves the display id back to the server's tool id, then POSTs to
        /tools/{id}/execute with a JSON body.
        """
        if not self._connected:
            await self.connect()
        
        # Resolve mapping; fall back to stripping prefix if necessary
        remote_id: Optional[str] = self._id_map.get(tool_id)
        if remote_id is None and self.tool_prefix and tool_id.startswith(f"{self.tool_prefix}_"):
            remote_id = tool_id[len(self.tool_prefix) + 1 :]
        if remote_id is None:
            # As a last resort, assume tool_id is already the remote id
            remote_id = tool_id
        
        try:
            if self.transport == MCPTransportType.HTTP:
                endpoint = f"{self.url}/tools/{remote_id}/execute"
                resp = await self._http_client.post(
                    endpoint,
                    json=parameters,
                )
                return resp.json()
            elif self.transport == MCPTransportType.WEBSOCKET:
                response = await self._send_websocket_message({
                    "jsonrpc": "2.0",
                    "id": f"exec_{remote_id}",
                    "method": "tools/call",
                    "params": {
                        "name": remote_id,
                        "arguments": parameters
                    }
                })
                if response and "result" in response:
                    return response["result"]
                elif response and "error" in response:
                    raise MCPError(f"Tool execution error: {response['error']}")
                else:
                    raise MCPError("Invalid response from MCP server")
            elif self.transport == MCPTransportType.STDIO:
                response = await self._send_stdio_message({
                    "jsonrpc": "2.0",
                    "id": f"exec_{remote_id}",
                    "method": "tools/call",
                    "params": {
                        "name": remote_id,
                        "arguments": parameters
                    }
                })
                if response and "result" in response:
                    return response["result"]
                elif response and "error" in response:
                    raise MCPError(f"Tool execution error: {response['error']}")
                else:
                    raise MCPError("Invalid response from MCP server")
            else:
                raise MCPError(f"Unsupported transport for tool execution: {self.transport}")
        except Exception as e:
            raise MCPError(f"Tool execution failed: {e}") from e
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()