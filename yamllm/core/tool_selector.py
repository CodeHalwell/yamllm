"""
Tool selection component for YAMLLM.

This module handles intelligent tool filtering and selection based on
prompt analysis and intent detection.
"""

from typing import Dict, List, Any, Optional
import re
import logging


class ToolSelector:
    """
    Handles intelligent tool filtering and selection.
    
    This class extracts tool selection logic from the main LLM class
    for better separation of concerns.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize tool selector.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger('yamllm.tool_selector')
    
    def filter_tools_for_prompt(
        self,
        tools: List[Dict[str, Any]],
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter tools based on prompt analysis.
        
        Args:
            tools: List of available tool definitions
            messages: List of messages (for prompt analysis)
            
        Returns:
            Filtered list of relevant tools
        """
        if not tools or not messages:
            return tools
        
        # Extract prompt text from last user message
        prompt_text = ""
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                prompt_text = msg.get('content', '')
                break
        
        if not prompt_text:
            return tools
        
        # Check for explicit tool request
        explicit_tool = self._extract_explicit_tool(prompt_text)
        if explicit_tool:
            # Return only the explicitly requested tool if it exists
            matching_tools = [t for t in tools if t.get('function', {}).get('name') == explicit_tool]
            if matching_tools:
                self.logger.debug(f"Using explicitly requested tool: {explicit_tool}")
                return matching_tools
        
        # Extract intents
        intents = self._extract_intent(prompt_text)
        
        # Filter tools based on intents
        filtered = []
        for tool in tools:
            tool_name = tool.get('function', {}).get('name', '')
            if self._tool_matches_intent(tool_name, intents):
                filtered.append(tool)
        
        # If no tools match, return all (let the model decide)
        if not filtered:
            self.logger.debug("No specific tools matched intent, providing all tools")
            return tools
        
        self.logger.debug(f"Filtered to {len(filtered)} relevant tools from {len(tools)}")
        return filtered
    
    def _extract_intent(self, prompt_text: str) -> Dict[str, bool]:
        """Extract lightweight intents from prompt to guide tool selection."""
        text = (prompt_text or "").lower()
        
        domain_hint = re.search(r"\b(?:[a-z0-9-]+\.)+(?:[a-z]{2,})(?:/[^\s]*)?\b", text)
        explicit_url = re.search(r"https?://[^\s]+", text)
        command_match = re.search(r"(?:use|call|run|invoke)\s+(?:the\s+)?([a-z0-9_\-]+)\s+tool", text)
        
        wants = {
            "web": any(k in text for k in ("search", "look up", "latest", "today", "current", "news", "headline", "website", "site", "scrape", "scraper", "crawl", "fetch"))
                    or bool(explicit_url) or bool(domain_hint),
            "calc": any(k in text for k in ("calculate", "calc", "sum", "difference", "multiply", "divide", "percent", "product"))
                    or bool(re.search(r"\b\d+\s*([\+\-\*/×÷])\s*\d+", text)),
            "convert": any(k in text for k in ("convert", "units", "unit", "km", "miles", "celsius", "fahrenheit")),
            "time": any(k in text for k in ("time in", "timezone", "utc", "pst", "est")),
            "files": any(k in text for k in ("read file", "open file", "path", "search file", "grep")),
            "url": bool(explicit_url) or bool(domain_hint),
            "csv": "csv" in text,
            "json": "json" in text and any(k in text for k in ("pretty", "minify", "validate")),
            "regex": "regex" in text or bool(re.search(r"\/[a-zA-Z0-9_\-\.\+\*\?\|\(\)\[\]\{\}]+\/", text)),
            "hash": "hash" in text or any(k in text for k in ("md5", "sha256", "sha1")),
            "base64": "base64" in text or any(k in text for k in ("encode", "decode")),
            "uuid": "uuid" in text,
            "direct_tool": bool(command_match),
            "random": any(k in text for k in ("random", "generate")),
            "weather": "weather" in text,
            "datetime": any(k in text for k in ("date", "time", "now", "today")),
            "lorem": "lorem" in text or "placeholder" in text,
        }
        
        # Expand web intent if explicit URL present
        if wants["url"]:
            wants["web"] = True
        
        return wants
    
    def _extract_explicit_tool(self, prompt_text: str) -> Optional[str]:
        """Extract explicitly requested tool name from prompt."""
        text = (prompt_text or "").lower()
        matches = re.findall(r"(?:use|call|run|invoke)\s+(?:the\s+)?([a-z0-9_\-]+)\s+tool", text)
        
        if not matches:
            # Check for bare usage patterns
            bare_match = re.search(r"\buse\s+(webscrape|webscraper|web_scraper|webscraping)\b", text)
            if bare_match:
                matches = [bare_match.group(1)]
        
        if not matches:
            return None
        
        # Map aliases to canonical names
        alias_map = {
            "webscrape": "web_scraper",
            "webscraper": "web_scraper",
            "web_scraper": "web_scraper",
            "webscraping": "web_scraper",
            "scrape": "web_scraper",
            "scraper": "web_scraper",
            "websearch": "web_search",
            "search": "web_search",
            "calc": "calculator",
        }
        
        requested = matches[-1].replace('-', '_')
        return alias_map.get(requested, requested)
    
    def _tool_matches_intent(self, tool_name: str, intents: Dict[str, bool]) -> bool:
        """Check if a tool matches the detected intents."""
        tool_name_lower = tool_name.lower()
        
        # Map tool names to intents
        tool_intent_map = {
            "web_search": ["web"],
            "web_scraper": ["web", "url"],
            "web_headlines": ["web", "news"],
            "url_metadata": ["web", "url"],
            "calculator": ["calc"],
            "unit_converter": ["convert"],
            "timezone": ["time"],
            "weather": ["weather"],
            "file_read": ["files"],
            "file_search": ["files"],
            "csv_preview": ["files", "csv"],
            "json_tool": ["json"],
            "regex_extract": ["regex"],
            "hash_tool": ["hash"],
            "base64_encode": ["base64"],
            "base64_decode": ["base64"],
            "uuid_tool": ["uuid"],
            "random_string": ["random"],
            "random_number": ["random"],
            "datetime": ["datetime", "time"],
            "lorem_ipsum": ["lorem"],
        }
        
        # Get intents for this tool
        tool_intents = tool_intent_map.get(tool_name_lower, [])
        
        # Check if any tool intent matches
        for tool_intent in tool_intents:
            if intents.get(tool_intent, False):
                return True
        
        return False
    
    def intent_requires_tools(self, prompt: str) -> bool:
        """
        Check if prompt likely requires tool use.
        
        Args:
            prompt: The prompt text
            
        Returns:
            True if tools are likely needed
        """
        intents = self._extract_intent(prompt)
        return any(intents.values())
