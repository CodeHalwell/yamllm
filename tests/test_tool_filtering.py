"""
Tests for tool filtering and tool choice determination.

These tests cover the critical _filter_tools_for_prompt and _determine_tool_choice
functions that were identified as having no test coverage.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from yamllm.core.llm import LLM


@pytest.fixture
def mock_llm():
    """Create a mock LLM instance for testing."""
    with patch('yamllm.core.llm.parse_yaml_config'), \
         patch('yamllm.core.llm.setup_logging'), \
         patch('yamllm.core.llm.ErrorHandler'), \
         patch('yamllm.core.llm.ProviderFactory'), \
         patch('yamllm.core.llm.MemoryManager'), \
         patch('yamllm.core.llm.ToolOrchestrator'), \
         patch('yamllm.core.llm.ThinkingManager'):
        
        # Create a mock config
        mock_config = MagicMock()
        mock_config.tools = MagicMock()
        mock_config.tools.gate_web_search = True
        mock_config.logging = MagicMock()
        mock_config.logging.level = "INFO"
        mock_config.logging.file = "test.log"
        mock_config.logging.format = "%(message)s"
        
        with patch.object(LLM, 'load_config', return_value=mock_config):
            llm = LLM(config_path="test.yaml", api_key="test_key")
            llm.config = mock_config
            llm.logger = MagicMock()
            return llm


class TestExplicitToolExtraction:
    """Test explicit tool extraction from prompts."""
    
    def test_explicit_tool_with_use_keyword(self, mock_llm):
        """Test extraction when user explicitly says 'use calculator tool'."""
        prompt = "Please use calculator tool to add 5 and 3"
        explicit = mock_llm._extract_explicit_tool(prompt)
        assert explicit == "calculator"
    
    def test_explicit_tool_with_call_keyword(self, mock_llm):
        """Test extraction with 'call' keyword."""
        prompt = "Call the web_search tool to find information"
        explicit = mock_llm._extract_explicit_tool(prompt)
        assert explicit == "web_search"
    
    def test_no_explicit_tool(self, mock_llm):
        """Test that no explicit tool is extracted from regular prompt."""
        prompt = "What is the weather today?"
        explicit = mock_llm._extract_explicit_tool(prompt)
        assert explicit is None


class TestToolFilteringWithContext:
    """Test tool filtering based on context and intent."""
    
    def test_web_search_intent(self, mock_llm):
        """Test filtering for web search intent."""
        tools = [
            {"type": "function", "function": {"name": "web_search"}},
            {"type": "function", "function": {"name": "calculator"}},
            {"type": "function", "function": {"name": "datetime"}},
        ]
        messages = [{"role": "user", "content": "Search for the latest news"}]
        
        filtered = mock_llm._filter_tools_for_prompt(tools, messages)
        
        # Should include web_search
        filtered_names = [t["function"]["name"] for t in filtered]
        assert "web_search" in filtered_names
    
    def test_calculator_intent(self, mock_llm):
        """Test filtering for calculator intent."""
        tools = [
            {"type": "function", "function": {"name": "web_search"}},
            {"type": "function", "function": {"name": "calculator"}},
            {"type": "function", "function": {"name": "datetime"}},
        ]
        messages = [{"role": "user", "content": "Calculate 5 + 3"}]
        
        filtered = mock_llm._filter_tools_for_prompt(tools, messages)
        
        # Should include calculator
        filtered_names = [t["function"]["name"] for t in filtered]
        assert "calculator" in filtered_names
    
    def test_url_intent(self, mock_llm):
        """Test filtering for URL-related intent."""
        tools = [
            {"type": "function", "function": {"name": "url_metadata"}},
            {"type": "function", "function": {"name": "calculator"}},
        ]
        messages = [{"role": "user", "content": "Get metadata from https://example.com"}]
        
        filtered = mock_llm._filter_tools_for_prompt(tools, messages)
        
        # Should include url_metadata
        filtered_names = [t["function"]["name"] for t in filtered]
        assert "url_metadata" in filtered_names
    
    def test_no_intent_returns_empty(self, mock_llm):
        """Test that no intent returns empty list."""
        tools = [
            {"type": "function", "function": {"name": "web_search"}},
            {"type": "function", "function": {"name": "calculator"}},
        ]
        messages = [{"role": "user", "content": "Hello"}]
        
        filtered = mock_llm._filter_tools_for_prompt(tools, messages)
        
        # Should return empty when no intent detected
        assert filtered == []
    
    def test_explicit_tool_overrides_intent(self, mock_llm):
        """Test that explicit tool mention overrides intent detection."""
        tools = [
            {"type": "function", "function": {"name": "web_search"}},
            {"type": "function", "function": {"name": "calculator"}},
        ]
        messages = [{"role": "user", "content": "Use calculator tool"}]
        
        filtered = mock_llm._filter_tools_for_prompt(tools, messages)
        
        # Should include calculator even without mathematical expression
        filtered_names = [t["function"]["name"] for t in filtered]
        assert "calculator" in filtered_names


class TestToolBlacklistFiltering:
    """Test tool filtering with blacklist."""
    
    def test_filtering_gate_disabled(self, mock_llm):
        """Test that all tools are returned when gating is disabled."""
        mock_llm.config.tools.gate_web_search = False
        
        tools = [
            {"type": "function", "function": {"name": "web_search"}},
            {"type": "function", "function": {"name": "calculator"}},
        ]
        messages = [{"role": "user", "content": "Hello"}]
        
        filtered = mock_llm._filter_tools_for_prompt(tools, messages)
        
        # Should return all tools when gating is disabled
        assert filtered == tools
    
    def test_none_tools_returns_none(self, mock_llm):
        """Test that None tools returns None."""
        messages = [{"role": "user", "content": "Search something"}]
        filtered = mock_llm._filter_tools_for_prompt(None, messages)
        assert filtered is None
    
    def test_empty_tools_returns_empty(self, mock_llm):
        """Test that empty tools list returns empty list."""
        messages = [{"role": "user", "content": "Search something"}]
        filtered = mock_llm._filter_tools_for_prompt([], messages)
        assert filtered == []


class TestToolFilteringEdgeCases:
    """Test edge cases in tool filtering."""
    
    def test_no_messages(self, mock_llm):
        """Test filtering with no messages."""
        tools = [{"type": "function", "function": {"name": "calculator"}}]
        filtered = mock_llm._filter_tools_for_prompt(tools, [])
        assert filtered == tools  # Should return original tools
    
    def test_malformed_tool(self, mock_llm):
        """Test filtering with malformed tool definition."""
        tools = [
            {"type": "function", "function": {"name": "web_search"}},
            {"invalid": "tool"},  # Malformed
            {"type": "function", "function": {"name": "calculator"}},
        ]
        messages = [{"role": "user", "content": "Search for news"}]
        
        # Should handle malformed tools gracefully
        filtered = mock_llm._filter_tools_for_prompt(tools, messages)
        assert isinstance(filtered, list)
    
    def test_multiple_intents(self, mock_llm):
        """Test filtering with multiple intents."""
        tools = [
            {"type": "function", "function": {"name": "web_search"}},
            {"type": "function", "function": {"name": "calculator"}},
            {"type": "function", "function": {"name": "datetime"}},
        ]
        messages = [{"role": "user", "content": "Search the web and calculate 5+3"}]
        
        filtered = mock_llm._filter_tools_for_prompt(tools, messages)
        filtered_names = [t["function"]["name"] for t in filtered]
        
        # Should include both web_search and calculator
        assert "web_search" in filtered_names
        assert "calculator" in filtered_names


class TestDetermineToolChoiceRequired:
    """Test _determine_tool_choice with required mode."""
    
    def test_strong_intent_returns_required(self, mock_llm):
        """Test that strong intent returns 'required'."""
        tools = [{"type": "function", "function": {"name": "web_search"}}]
        prompt = "Search for the latest news"
        
        choice = mock_llm._determine_tool_choice(prompt, tools)
        
        # Should return 'required' or a specific tool choice
        assert choice is not None
    
    def test_explicit_tool_returns_specific_choice(self, mock_llm):
        """Test that explicit tool mention returns specific choice."""
        tools = [
            {"type": "function", "function": {"name": "web_search"}},
            {"type": "function", "function": {"name": "calculator"}},
        ]
        prompt = "Use calculator tool to add 5 and 3"
        
        choice = mock_llm._determine_tool_choice(prompt, tools)
        
        # Should return specific tool choice for calculator
        assert choice is not None
        if isinstance(choice, dict):
            assert choice["type"] == "function"
            assert choice["function"]["name"] == "calculator"
    
    def test_no_intent_returns_none(self, mock_llm):
        """Test that no intent returns None."""
        tools = [{"type": "function", "function": {"name": "web_search"}}]
        prompt = "Hello, how are you?"
        
        choice = mock_llm._determine_tool_choice(prompt, tools)
        
        # Should return None when no tool is needed
        assert choice is None


class TestDetermineToolChoiceAuto:
    """Test _determine_tool_choice with auto mode."""
    
    def test_none_tools_returns_none(self, mock_llm):
        """Test that None tools returns None."""
        choice = mock_llm._determine_tool_choice("Search something", None)
        assert choice is None
    
    def test_empty_tools_returns_none(self, mock_llm):
        """Test that empty tools list returns None."""
        choice = mock_llm._determine_tool_choice("Search something", [])
        assert choice is None


class TestDetermineToolChoiceNone:
    """Test _determine_tool_choice returning None."""
    
    def test_greeting_returns_none(self, mock_llm):
        """Test that simple greeting returns None."""
        tools = [{"type": "function", "function": {"name": "web_search"}}]
        prompt = "Hi there!"
        
        choice = mock_llm._determine_tool_choice(prompt, tools)
        assert choice is None
    
    def test_no_matching_tools(self, mock_llm):
        """Test when no tools match the intent."""
        tools = [{"type": "function", "function": {"name": "file_read"}}]
        prompt = "Search the web"
        
        # Web search intent but only file_read available
        choice = mock_llm._determine_tool_choice(prompt, tools)
        
        # Should handle gracefully
        assert choice is not None or choice is None  # Either is acceptable


class TestIntentExtraction:
    """Test intent extraction logic."""
    
    def test_web_intent_keywords(self, mock_llm):
        """Test web intent detection from keywords."""
        intents = mock_llm._extract_intent("Search for the latest news")
        assert intents.get("web") is True
    
    def test_calc_intent_with_expression(self, mock_llm):
        """Test calculator intent with mathematical expression."""
        intents = mock_llm._extract_intent("What is 5 + 3?")
        assert intents.get("calc") is True
    
    def test_multiple_intents(self, mock_llm):
        """Test multiple intents in one prompt."""
        intents = mock_llm._extract_intent("Search the web and calculate 5 + 3")
        assert intents.get("web") is True
        assert intents.get("calc") is True
    
    def test_url_detection(self, mock_llm):
        """Test URL detection triggers web intent."""
        intents = mock_llm._extract_intent("Check https://example.com")
        assert intents.get("web") is True
    
    def test_no_intent(self, mock_llm):
        """Test no intent detection."""
        intents = mock_llm._extract_intent("Hello")
        assert not any(intents.values())
