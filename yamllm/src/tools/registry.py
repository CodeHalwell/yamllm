from typing import Dict, Type
from .base import Tool

class ToolRegistry:
    _instance = None
    _tools: Dict[str, Tool] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def register_tool(self, tool: Tool) -> None:
        """Register a new tool instance"""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Tool:
        """Retrieve a tool by name"""
        return self._tools.get(name)

    def list_tools(self) -> Dict[str, Dict]:
        """Return all registered tools and their signatures"""
        return {name: tool.get_signature() for name, tool in self._tools.items()}
