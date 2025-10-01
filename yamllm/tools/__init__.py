from .utility_tools import WebSearch, Calculator, TimezoneTool, UnitConverter, WeatherTool, WebScraper
from .base import Tool, ToolRegistry
from .manager import ToolExecutor

# Backward compatibility alias
ToolManager = ToolExecutor

__all__ = [
    'Tool',
    'ToolRegistry',
    'ToolExecutor',
    'ToolManager',  # Backward compatibility
    'WebSearch',
    'Calculator',
    'TimezoneTool',
    'UnitConverter',
    'WeatherTool',
    'WebScraper'
]
