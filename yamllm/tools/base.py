from typing import Dict, Any, List, Optional

class Tool:
    def __init__(self, name: Optional[str] = None, description: Optional[str] = None):
        # Allow subclasses to define class attributes for name/description
        self.name = name or getattr(self, "name", None)
        self.description = description or getattr(self, "description", None)
        if not self.name or not self.description:
            raise TypeError("Tool requires 'name' and 'description'")

    def execute(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Tool must implement execute method")

    def _get_parameters(self) -> Dict:
        """
        Return a JSON Schema parameters object describing the tool arguments.
        Subclasses should override this.
        """
        return {"type": "object", "properties": {}}

    def get_signature(self) -> Dict[str, Any]:
        """Return provider-friendly tool definition (OpenAI function schema)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._get_parameters(),
            },
        }

class ToolRegistry:
    """
    Deprecated: Use ToolManager instead. Kept for backward compatibility in docs.
    """
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry")
        return self._tools[name]

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())
