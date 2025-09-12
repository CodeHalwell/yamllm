"""Tests for type safety and response models."""

import pytest
from unittest.mock import Mock

from yamllm.core.models import (
    CompletionResponse, 
    ToolCall, 
    ToolFunction, 
    Usage,
    ResponseAdapter,
    ProviderResponseValidator
)
from yamllm.core.exceptions import ToolExecutionError
from yamllm.tools.manager import ToolManager
from yamllm.tools.base import Tool


class MockTool(Tool):
    """Mock tool for testing."""
    
    def __init__(self, should_fail=False, fail_with=None):
        super().__init__("mock_tool", "Mock tool for testing")
        self.should_fail = should_fail
        self.fail_with = fail_with
    
    def execute(self, **kwargs):
        if self.should_fail:
            if self.fail_with:
                raise self.fail_with
            else:
                raise ValueError("Mock failure")
        return {"result": "success", "args": kwargs}
    
    def _get_parameters(self):
        return {
            "type": "object",
            "properties": {
                "test_param": {"type": "string"}
            }
        }


class TestResponseModels:
    """Test response model validation and adaptation."""
    
    def test_completion_response_basic(self):
        """Test basic completion response creation."""
        response = CompletionResponse(
            content="Hello world",
            model="test-model",
            provider="test"
        )
        
        assert response.content == "Hello world"
        assert response.model == "test-model"
        assert response.provider == "test"
        assert response.tool_calls is None
    
    def test_completion_response_with_tools(self):
        """Test completion response with tool calls."""
        tool_call = ToolCall(
            id="call_123",
            type="function",
            function=ToolFunction(
                name="test_function",
                arguments='{"param": "value"}'
            )
        )
        
        response = CompletionResponse(
            content="Using tool...",
            tool_calls=[tool_call],
            provider="test"
        )
        
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].id == "call_123"
        assert response.tool_calls[0].function.name == "test_function"
        
        # Test argument parsing
        args = response.tool_calls[0].function.get_arguments_dict()
        assert args == {"param": "value"}
    
    def test_usage_calculation(self):
        """Test automatic total token calculation."""
        usage = Usage(prompt_tokens=10, completion_tokens=5)
        assert usage.total_tokens == 15
        
        # Test explicit total
        usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=20)
        assert usage.total_tokens == 20
    
    def test_openai_response_adapter(self):
        """Test OpenAI response adaptation."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Hello from OpenAI"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4"
        mock_response.usage = Mock()
        mock_response.usage.model_dump.return_value = {
            "prompt_tokens": 10, 
            "completion_tokens": 5, 
            "total_tokens": 15
        }
        
        adapted = ResponseAdapter.adapt_openai_response(mock_response)
        
        assert isinstance(adapted, CompletionResponse)
        assert adapted.content == "Hello from OpenAI"
        assert adapted.model == "gpt-4"
        assert adapted.provider == "openai"
        assert adapted.finish_reason == "stop"
        assert adapted.usage.total_tokens == 15
    
    def test_anthropic_response_adapter(self):
        """Test Anthropic response adaptation."""
        mock_response = {
            "content": [
                {"type": "text", "text": "Hello from Claude"}
            ],
            "model": "claude-3-sonnet",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 12,
                "output_tokens": 8
            }
        }
        
        adapted = ResponseAdapter.adapt_anthropic_response(mock_response)
        
        assert isinstance(adapted, CompletionResponse)
        assert adapted.content == "Hello from Claude"
        assert adapted.model == "claude-3-sonnet"
        assert adapted.provider == "anthropic"
        assert adapted.finish_reason == "end_turn"
        assert adapted.usage.prompt_tokens == 12
        assert adapted.usage.completion_tokens == 8
    
    def test_provider_response_validator(self):
        """Test provider response validation."""
        # Test None response
        with pytest.raises(ValueError, match="Received None response"):
            ProviderResponseValidator.validate_and_adapt(None, "test")
        
        # Test generic response
        response = {"content": "Generic response"}
        adapted = ProviderResponseValidator.validate_and_adapt(response, "generic")
        
        assert adapted.content == "Generic response"
        assert adapted.provider == "generic"


class TestToolExceptionHandling:
    """Test improved tool exception handling."""
    
    def test_tool_manager_timeout(self):
        """Test tool timeout handling."""
        slow_tool = MockTool()
        # Mock slow execution
        import time
        original_execute = slow_tool.execute
        def slow_execute(**kwargs):
            time.sleep(0.1)  # Longer than timeout
            return original_execute(**kwargs)
        slow_tool.execute = slow_execute
        
        manager = ToolManager(timeout=0.05)  # 50ms timeout
        manager.register(slow_tool)
        
        with pytest.raises(ToolExecutionError, match="timed out"):
            manager.execute("mock_tool", {})
    
    def test_tool_manager_missing_argument(self):
        """Test missing argument handling."""
        tool = MockTool()
        # Override to require argument
        def strict_execute(test_param, **kwargs):
            return {"result": f"Got {test_param}"}
        tool.execute = strict_execute
        
        manager = ToolManager()
        manager.register(tool)
        
        with pytest.raises(ToolExecutionError, match="Invalid argument type or value"):
            manager.execute("mock_tool", {})  # Missing test_param
    
    def test_tool_manager_value_error(self):
        """Test ValueError handling."""
        tool = MockTool(should_fail=True, fail_with=ValueError("Invalid value"))
        
        manager = ToolManager()
        manager.register(tool)
        
        with pytest.raises(ToolExecutionError, match="Invalid argument type or value"):
            manager.execute("mock_tool", {})
    
    def test_tool_manager_generic_error(self):
        """Test generic exception handling."""
        tool = MockTool(should_fail=True, fail_with=RuntimeError("Something went wrong"))
        
        manager = ToolManager()
        manager.register(tool)
        
        with pytest.raises(ToolExecutionError, match="Unexpected error"):
            manager.execute("mock_tool", {})
    
    def test_successful_tool_execution(self):
        """Test successful tool execution."""
        tool = MockTool()
        
        manager = ToolManager()
        manager.register(tool)
        
        result = manager.execute("mock_tool", {"test_param": "value"})
        assert result["result"] == "success"
        assert result["args"]["test_param"] == "value"