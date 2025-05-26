"""
Model Context Protocol (MCP) client for YAMLLM.

This module provides the client implementation for interacting with MCP servers.
MCP allows language models to interact with external tools and services in a standardized way.
"""

from .client import MCPClient
from .connector import MCPConnector

__all__ = [
    "MCPClient",
    "MCPConnector"
]