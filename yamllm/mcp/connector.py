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
        
    def _process_auth(self, auth_str):
        """Process authentication string, resolving environment variables."""
        if not auth_str:
            return None
            
        env_var_match = re.match(r"\${(.+)}", auth_str)
        if env_var_match:
            env_var_name = env_var_match.group(1)
            return os.environ.get(env_var_name)
        return auth_str
