"""
MCP Connector implementation for YAMLLM.
"""

import os
import re
import logging
import requests


class MCPConnector:
    """Connector for MCP servers."""
    
    def __init__(self, name, url, authentication=None, description=None, tool_prefix=None):
        self.name = name
        self.url = url.rstrip("/")
        self.description = description or f"MCP connector for {name}"
        self.tool_prefix = tool_prefix
        self.auth_token = self._process_auth(authentication)
        self.logger = logging.getLogger(__name__)
        self._cached_tools = None
