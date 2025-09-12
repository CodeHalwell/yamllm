"""
Enhanced tool management system for YAMLLM.

This module provides advanced tool management capabilities including:
- Tool discovery and installation
- Tool testing and validation
- Tool pack management
- Tool performance monitoring
- Tool configuration and customization
"""

import os
import json
import importlib
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

from rich.console import Console

console = Console()


@dataclass
class ToolInfo:
    """Information about a tool."""
    name: str
    description: str
    category: str
    version: str = "1.0.0"
    author: str = "Unknown"
    requires_api_key: bool = False
    api_key_env: Optional[str] = None
    dependencies: List[str] = None
    enabled: bool = True
    usage_count: int = 0
    last_used: Optional[datetime] = None
    average_execution_time: float = 0.0
    success_rate: float = 100.0
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class ToolPack:
    """Information about a tool pack."""
    name: str
    description: str
    tools: List[str]
    category: str = "general"
    version: str = "1.0.0"
    enabled: bool = True


class ToolRegistry:
    """Registry for managing tool information and metadata."""
    
    def __init__(self):
        self.tools: Dict[str, ToolInfo] = {}
        self.packs: Dict[str, ToolPack] = {}
        self._load_built_in_tools()
        self._load_built_in_packs()
    
    def _load_built_in_tools(self):
        """Load information about built-in tools."""
        built_in_tools = {
            "calculator": ToolInfo(
                name="calculator",
                description="Perform mathematical calculations and expressions",
                category="utility",
                author="YAMLLM Team"
            ),
            "weather": ToolInfo(
                name="weather",
                description="Get current weather information for any location",
                category="information",
                requires_api_key=True,
                api_key_env="WEATHER_API_KEY",
                author="YAMLLM Team"
            ),
            "web_search": ToolInfo(
                name="web_search",
                description="Search the web using DuckDuckGo",
                category="information",
                author="YAMLLM Team"
            ),
            "web_scraper": ToolInfo(
                name="web_scraper",
                description="Extract content from web pages",
                category="information",
                author="YAMLLM Team"
            ),
            "unit_converter": ToolInfo(
                name="unit_converter",
                description="Convert between different units of measurement",
                category="utility",
                author="YAMLLM Team"
            ),
            "timezone": ToolInfo(
                name="timezone",
                description="Get timezone information and convert times",
                category="utility",
                author="YAMLLM Team"
            ),
            "datetime": ToolInfo(
                name="datetime",
                description="Get current date and time information",
                category="utility",
                author="YAMLLM Team"
            ),
            "file_reader": ToolInfo(
                name="file_reader",
                description="Read and analyze file contents",
                category="files",
                author="YAMLLM Team"
            ),
            "file_search": ToolInfo(
                name="file_search",
                description="Search for files and directories",
                category="files",
                author="YAMLLM Team"
            ),
            "csv_preview": ToolInfo(
                name="csv_preview",
                description="Preview and analyze CSV files",
                category="files",
                author="YAMLLM Team"
            ),
            "base64_encode": ToolInfo(
                name="base64_encode",
                description="Encode text to Base64",
                category="text",
                author="YAMLLM Team"
            ),
            "base64_decode": ToolInfo(
                name="base64_decode", 
                description="Decode Base64 text",
                category="text",
                author="YAMLLM Team"
            ),
            "hash_tool": ToolInfo(
                name="hash_tool",
                description="Generate various hash values (MD5, SHA256, etc.)",
                category="text",
                author="YAMLLM Team"
            ),
            "json_tool": ToolInfo(
                name="json_tool",
                description="Parse, validate, and format JSON",
                category="text",
                author="YAMLLM Team"
            ),
            "regex_extract": ToolInfo(
                name="regex_extract",
                description="Extract data using regular expressions",
                category="text",
                author="YAMLLM Team"
            ),
            "random_string": ToolInfo(
                name="random_string",
                description="Generate random strings",
                category="utility",
                author="YAMLLM Team"
            ),
            "random_number": ToolInfo(
                name="random_number",
                description="Generate random numbers",
                category="utility",
                author="YAMLLM Team"
            ),
            "lorem_ipsum": ToolInfo(
                name="lorem_ipsum",
                description="Generate Lorem Ipsum placeholder text",
                category="text",
                author="YAMLLM Team"
            ),
            "uuid_generator": ToolInfo(
                name="uuid_generator",
                description="Generate UUIDs",
                category="utility",
                author="YAMLLM Team"
            ),
            "url_metadata": ToolInfo(
                name="url_metadata",
                description="Extract metadata from URLs",
                category="information",
                author="YAMLLM Team"
            ),
            "web_headlines": ToolInfo(
                name="web_headlines",
                description="Get news headlines from websites",
                category="information",
                author="YAMLLM Team"
            ),
            "tools_help": ToolInfo(
                name="tools_help",
                description="Get detailed help information about tools",
                category="utility",
                author="YAMLLM Team"
            )
        }
        
        self.tools.update(built_in_tools)
    
    def _load_built_in_packs(self):
        """Load information about built-in tool packs."""
        built_in_packs = {
            "common": ToolPack(
                name="common",
                description="Essential tools for everyday use",
                tools=["calculator", "weather", "timezone", "datetime", "uuid_generator"],
                category="essential"
            ),
            "web": ToolPack(
                name="web",
                description="Web browsing and search tools",
                tools=["web_search", "web_scraper", "url_metadata", "web_headlines"],
                category="information"
            ),
            "files": ToolPack(
                name="files",
                description="File manipulation and reading tools",
                tools=["file_reader", "file_search", "csv_preview"],
                category="productivity"
            ),
            "text": ToolPack(
                name="text",
                description="Text processing and manipulation",
                tools=["base64_encode", "base64_decode", "hash_tool", "json_tool", "regex_extract"],
                category="text_processing"
            ),
            "utility": ToolPack(
                name="utility",
                description="Utility functions and generators",
                tools=["random_string", "random_number", "lorem_ipsum", "unit_converter"],
                category="utility"
            )
        }
        
        self.packs.update(built_in_packs)
    
    def get_tool(self, name: str) -> Optional[ToolInfo]:
        """Get tool information by name."""
        return self.tools.get(name)
    
    def get_pack(self, name: str) -> Optional[ToolPack]:
        """Get pack information by name."""
        return self.packs.get(name)
    
    def list_tools(self, category: Optional[str] = None) -> List[ToolInfo]:
        """List tools, optionally filtered by category."""
        tools = list(self.tools.values())
        if category:
            tools = [t for t in tools if t.category == category]
        return sorted(tools, key=lambda x: x.name)
    
    def list_packs(self, category: Optional[str] = None) -> List[ToolPack]:
        """List packs, optionally filtered by category."""
        packs = list(self.packs.values())
        if category:
            packs = [p for p in packs if p.category == category]
        return sorted(packs, key=lambda x: x.name)
    
    def search_tools(self, query: str) -> List[ToolInfo]:
        """Search tools by name or description."""
        query = query.lower()
        results = []
        for tool in self.tools.values():
            if (query in tool.name.lower() or 
                query in tool.description.lower() or
                query in tool.category.lower()):
                results.append(tool)
        return sorted(results, key=lambda x: x.name)
    
    def get_categories(self) -> List[str]:
        """Get all available tool categories."""
        categories = set(tool.category for tool in self.tools.values())
        return sorted(list(categories))


class ToolRegistryManager:
    """Enhanced tool metadata/registry manager (CLI-facing)."""
    
    def __init__(self):
        self.registry = ToolRegistry()
        self.stats: Dict[str, Dict[str, Any]] = {}
        self._load_stats()
    
    def _load_stats(self):
        """Load tool usage statistics."""
        stats_file = Path.home() / ".yamllm" / "tool_stats.json"
        if stats_file.exists():
            try:
                with open(stats_file, 'r') as f:
                    self.stats = json.load(f)
            except Exception:
                self.stats = {}
    
    def _save_stats(self):
        """Save tool usage statistics."""
        stats_dir = Path.home() / ".yamllm"
        stats_dir.mkdir(parents=True, exist_ok=True)
        
        stats_file = stats_dir / "tool_stats.json"
        try:
            with open(stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2, default=str)
        except Exception:
            pass
    
    def record_tool_usage(self, tool_name: str, execution_time: float, success: bool):
        """Record tool usage statistics."""
        if tool_name not in self.stats:
            self.stats[tool_name] = {
                "usage_count": 0,
                "total_execution_time": 0.0,
                "success_count": 0,
                "last_used": None
            }
        
        stats = self.stats[tool_name]
        stats["usage_count"] += 1
        stats["total_execution_time"] += execution_time
        if success:
            stats["success_count"] += 1
        stats["last_used"] = datetime.now().isoformat()
        
        # Update registry info
        tool_info = self.registry.get_tool(tool_name)
        if tool_info:
            tool_info.usage_count = stats["usage_count"]
            tool_info.average_execution_time = stats["total_execution_time"] / stats["usage_count"]
            tool_info.success_rate = (stats["success_count"] / stats["usage_count"]) * 100
            tool_info.last_used = datetime.fromisoformat(stats["last_used"])
        
        self._save_stats()
    
    def list_tools_detailed(self, category: Optional[str] = None, enabled_only: bool = False) -> List[ToolInfo]:
        """List tools with detailed information."""
        tools = self.registry.list_tools(category)
        if enabled_only:
            tools = [t for t in tools if t.enabled]
        return tools
    
    def get_tool_info(self, name: str) -> Optional[ToolInfo]:
        """Get detailed information about a tool."""
        return self.registry.get_tool(name)
    
    def test_tool(self, name: str) -> Dict[str, Any]:
        """Test a tool and return results."""
        tool_info = self.registry.get_tool(name)
        if not tool_info:
            return {"success": False, "error": f"Tool '{name}' not found"}
        
        result = {
            "name": name,
            "success": True,
            "error": None,
            "warnings": [],
            "requirements_met": True
        }
        
        # Check API key requirements
        if tool_info.requires_api_key and tool_info.api_key_env:
            if not os.getenv(tool_info.api_key_env):
                result["success"] = False
                result["error"] = f"Missing required environment variable: {tool_info.api_key_env}"
                result["requirements_met"] = False
        
        # Check dependencies
        for dep in tool_info.dependencies:
            try:
                importlib.import_module(dep)
            except ImportError:
                result["warnings"].append(f"Optional dependency missing: {dep}")
        
        return result
    
    def enable_tool(self, name: str) -> bool:
        """Enable a tool."""
        tool_info = self.registry.get_tool(name)
        if tool_info:
            tool_info.enabled = True
            return True
        return False
    
    def disable_tool(self, name: str) -> bool:
        """Disable a tool."""
        tool_info = self.registry.get_tool(name)
        if tool_info:
            tool_info.enabled = False
            return True
        return False
    
    def get_pack_tools(self, pack_name: str) -> List[ToolInfo]:
        """Get all tools in a pack."""
        pack = self.registry.get_pack(pack_name)
        if not pack:
            return []
        
        tools = []
        for tool_name in pack.tools:
            tool_info = self.registry.get_tool(tool_name)
            if tool_info:
                tools.append(tool_info)
        
        return tools
    
    def validate_configuration(self, config_tools: List[str], config_packs: List[str]) -> Dict[str, Any]:
        """Validate tool configuration."""
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "missing_tools": [],
            "missing_packs": [],
            "api_key_issues": []
        }
        
        # Check individual tools
        for tool_name in config_tools:
            if tool_name not in self.registry.tools:
                result["missing_tools"].append(tool_name)
                result["valid"] = False
        
        # Check tool packs
        for pack_name in config_packs:
            if pack_name not in self.registry.packs:
                result["missing_packs"].append(pack_name)
                result["valid"] = False
        
        # Check API key requirements
        all_tools = set(config_tools)
        for pack_name in config_packs:
            pack = self.registry.get_pack(pack_name)
            if pack:
                all_tools.update(pack.tools)
        
        for tool_name in all_tools:
            tool_info = self.registry.get_tool(tool_name)
            if tool_info and tool_info.requires_api_key and tool_info.api_key_env:
                if not os.getenv(tool_info.api_key_env):
                    result["api_key_issues"].append({
                        "tool": tool_name,
                        "env_var": tool_info.api_key_env
                    })
        
        if result["missing_tools"] or result["missing_packs"]:
            result["errors"].append("Some tools or packs not found")
        
        if result["api_key_issues"]:
            result["warnings"].append("Some tools require API keys that are not set")
        
        return result


# Global instance for CLI usage
tool_manager = ToolRegistryManager()
