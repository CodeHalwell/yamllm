import pytest

from yamllm.mcp.client import MCPClient
from yamllm.providers.base import ToolDefinition


class FakeAsyncConnector:
    def __init__(self, name: str, tools=None, exec_result=None):
        self.name = name
        self._tools = tools or []
        self._exec_result = exec_result or {"ok": True}
        self._discovered = False

    async def discover_tools(self, force_refresh: bool = False):
        self._discovered = True
        return self._tools

    async def execute_tool(self, tool_id: str, parameters):
        return self._exec_result

    async def disconnect(self):
        return None


@pytest.mark.anyio
async def test_mcp_client_discover_all_tools_async():
    client = MCPClient()
    tools = [
        {"name": "alpha", "description": "A", "parameters": {"type": "object", "properties": {}}},
        {"name": "beta", "description": "B", "parameters": {"type": "object", "properties": {}}},
    ]
    conn = FakeAsyncConnector("conn1", tools=tools)
    client.register_connector(conn)

    out = await client.discover_all_tools()
    assert "conn1" in out
    assert len(out["conn1"]) == 2
    assert out["conn1"][0]["name"] == "alpha"


@pytest.mark.anyio
async def test_mcp_client_execute_tool_async():
    client = MCPClient()
    conn = FakeAsyncConnector("connX", exec_result={"result": 42})
    client.register_connector(conn)

    res = await client.execute_tool("connX", "tool_id", {"x": 1})
    assert res == {"result": 42}


@pytest.mark.anyio
async def test_convert_mcp_tools_to_definitions_async():
    client = MCPClient()
    tools = [
        {"name": "gamma", "description": "G", "parameters": {"type": "object", "properties": {}}},
    ]
    conn = FakeAsyncConnector("srv", tools=tools)
    client.register_connector(conn)

    defs = await client.convert_mcp_tools_to_definitions()
    assert len(defs) == 1
    td = defs[0]
    assert isinstance(td, ToolDefinition)
    assert td.name == "gamma"
    assert td.description == "G"
    assert td.mcp_connector_name == "srv"
    assert td.mcp_tool_id == "gamma"
@pytest.fixture
def anyio_backend():
    return "asyncio"
