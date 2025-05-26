# Model Context Protocol (MCP) Support

YAMLLM supports the Model Context Protocol (MCP), which allows language models to interact with external tools and services in a standardized way.

## What is MCP?

The Model Context Protocol (MCP) is a standardized protocol for language models to discover and use tools from external servers. MCP servers expose a REST API that YAMLLM can interact with to:

1. Discover available tools
2. Execute tools with parameters
3. Return results to the language model

## Configuring MCP Connectors

You can configure MCP connectors in your YAML configuration file:

```yaml
tools:
  enabled: true
  tool_timeout: 30
  tool_list:
    - "web_search"
    - "calculator"
  mcp_connectors:
    - name: "zapier"
      url: "https://api.zapier.com/v1/mcp"
      authentication: "${ZAPIER_API_KEY}"
      description: "Zapier MCP connector"
      tool_prefix: "zapier"
      enabled: true
    - name: "custom_tools"
      url: "https://myserver.example.com/mcp"
      authentication: "${CUSTOM_MCP_KEY}"
      description: "Custom MCP tools"
      enabled: true
```

### MCP Connector Settings

- `name`: A unique name for the MCP connector within YAMLLM.
- `url`: The endpoint URL of the MCP server.
- `authentication`: API key or token for authenticating with the MCP server (can reference an environment variable with `${VAR_NAME}`).
- `description` (optional): User-friendly description.
- `tool_prefix` (optional): A prefix to add to tool names from this MCP server to avoid naming conflicts.
- `enabled` (optional): Whether this connector is enabled (default: true).

## How MCP Tools Work

When you configure MCP connectors, YAMLLM will:

1. Connect to the MCP servers to discover available tools
2. Make these tools available to the language model alongside local tools
3. Execute MCP tools when requested by the language model
4. Return the results to the language model

The language model can use MCP tools just like local tools, with no special syntax required.

## Security Considerations

- MCP servers have access to your data as it passes through them during tool execution.
- Store API keys in environment variables rather than directly in configuration files.
- Only connect to trusted MCP servers from trusted sources.

## Creating Your Own MCP Server

You can create your own MCP server that implements the MCP protocol:

- GET /tools - Returns a list of available tools
- POST /tools/{tool_id}/execute - Executes a tool with the given parameters

Tools should be defined with:
- A name
- A description
- A parameters schema (JSON Schema format)

For more information on implementing MCP servers, see the MCP specification.