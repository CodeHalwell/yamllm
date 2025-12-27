"""Tests for dynamic tool creation."""

from unittest.mock import Mock
from yamllm.tools.dynamic_tool_creator import (
    ToolCreator,
    ToolValidator,
    DynamicTool,
    ToolSpecification
)


def test_tool_validator_valid_code():
    """Test validation of safe code."""
    code = """
def execute(**kwargs):
    import json
    result = kwargs.get('value', 0) * 2
    return {"result": result}
"""

    is_valid, error = ToolValidator.validate_code(code)
    assert is_valid
    assert error is None


def test_tool_validator_forbidden_import():
    """Test rejection of forbidden imports."""
    code = """
def execute(**kwargs):
    import subprocess
    return subprocess.run(['ls'])
"""

    is_valid, error = ToolValidator.validate_code(code)
    assert not is_valid
    assert "subprocess" in error.lower()


def test_tool_validator_forbidden_eval():
    """Test rejection of eval/exec."""
    code = """
def execute(**kwargs):
    return eval(kwargs['code'])
"""

    is_valid, error = ToolValidator.validate_code(code)
    assert not is_valid
    assert "eval" in error.lower()


def test_tool_validator_syntax_error():
    """Test handling of syntax errors."""
    code = "def execute(**kwargs):\n    return {"

    is_valid, error = ToolValidator.validate_code(code)
    assert not is_valid
    assert "syntax" in error.lower()


def test_dynamic_tool_creation():
    """Test dynamic tool creation."""
    def test_func(**kwargs):
        return {"result": kwargs.get("value", 0) * 2}

    tool = DynamicTool(
        name="test_tool",
        description="Test tool",
        parameters={
            "type": "object",
            "properties": {
                "value": {"type": "number"}
            }
        },
        execute_func=test_func
    )

    assert tool.name == "test_tool"
    assert tool.description == "Test tool"

    # Test execution
    result = tool.execute(value=5)
    assert result == {"result": 10}


def test_dynamic_tool_get_schema():
    """Test tool schema generation."""
    def test_func(**kwargs):
        return {}

    tool = DynamicTool(
        name="test_tool",
        description="Test description",
        parameters={
            "type": "object",
            "properties": {
                "param": {"type": "string"}
            }
        },
        execute_func=test_func
    )

    schema = tool.get_schema()

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "test_tool"
    assert schema["function"]["description"] == "Test description"
    assert "parameters" in schema["function"]


def test_dynamic_tool_execution_error():
    """Test handling of execution errors."""
    def failing_func(**kwargs):
        raise ValueError("Test error")

    tool = DynamicTool(
        name="failing_tool",
        description="Test",
        parameters={},
        execute_func=failing_func
    )

    result = tool.execute()
    assert "error" in result
    assert "Test error" in result["error"]


def test_tool_creator_initialization():
    """Test tool creator initialization."""
    mock_llm = Mock()
    creator = ToolCreator(mock_llm)

    assert creator.llm == mock_llm
    assert creator.created_tools == {}


def test_tool_spec_dataclass():
    """Test ToolSpecification dataclass."""
    spec = ToolSpecification(
        name="test",
        description="Test tool",
        parameters={},
        code="def execute(**kwargs): return {}"
    )

    assert spec.name == "test"
    assert spec.description == "Test tool"
    assert spec.parameters == {}
    assert "execute" in spec.code
    assert spec.example_usage is None
    assert spec.safety_notes is None


def test_allowed_imports():
    """Test that allowed imports work."""
    allowed_modules = ["json", "re", "math", "datetime", "time"]

    for module in allowed_modules:
        code = f"""
import {module}
def execute(**kwargs):
    return {{"success": True}}
"""
        is_valid, error = ToolValidator.validate_code(code)
        assert is_valid, f"Module {module} should be allowed but got error: {error}"


def test_validator_attribute_string():
    """Test attribute string extraction."""
    import ast

    code = "os.system('ls')"
    tree = ast.parse(code)

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            attr_str = ToolValidator._get_attribute_string(node)
            assert attr_str == "os.system"


def test_tool_complexity():
    """Test creation of tools with various complexity levels."""
    # Simple tool
    simple_code = """
def execute(**kwargs):
    return {"result": "simple"}
"""

    is_valid, _ = ToolValidator.validate_code(simple_code)
    assert is_valid

    # Tool with logic
    complex_code = """
import json
import re

def execute(**kwargs):
    text = kwargs.get('text', '')
    pattern = kwargs.get('pattern', r'\\w+')

    matches = re.findall(pattern, text)
    result = {
        "matches": matches,
        "count": len(matches)
    }

    return result
"""

    is_valid, _ = ToolValidator.validate_code(complex_code)
    assert is_valid
