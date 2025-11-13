from .utility_tools import WebSearch, Calculator, TimezoneTool, UnitConverter, WeatherTool, WebScraper
from .base import Tool, ToolRegistry
from .manager import ToolExecutor
from .git_tools import (
    GitStatusTool, GitDiffTool, GitLogTool, GitBranchTool,
    GitCommitTool, GitPushTool, GitPullTool, GitError
)

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
    'WebScraper',
    'GitStatusTool',
    'GitDiffTool',
    'GitLogTool',
    'GitBranchTool',
    'GitCommitTool',
    'GitPushTool',
    'GitPullTool',
    'GitError'
]
