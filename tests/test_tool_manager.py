import time
import pytest
from yamllm.core.exceptions import ToolExecutionError
from typing import Dict, Any

from yamllm.tools.base import Tool
from yamllm.tools.manager import ToolManager


class SleepTool(Tool):
    def __init__(self, delay: float):
        super().__init__(name="sleep_tool", description="sleep for delay seconds")
        self.delay = delay

    def execute(self, delay: float) -> str:
        time.sleep(delay)
        return "ok"

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "delay": {"type": "number"},
            },
            "required": ["delay"],
        }


class ErrorTool(Tool):
    def __init__(self):
        super().__init__(name="error_tool", description="always raises error")

    def execute(self) -> str:
        raise ValueError("boom")


def test_tool_manager_timeout():
    tm = ToolManager(timeout=0.05)
    tm.register(SleepTool(delay=1.0))
    with pytest.raises(ToolExecutionError, match="timed out"):
        tm.execute("sleep_tool", {"delay": 0.2})


def test_tool_manager_error_normalization():
    tm = ToolManager(timeout=1)
    tm.register(ErrorTool())
    with pytest.raises(ToolExecutionError, match="Invalid argument type or value"):
        tm.execute("error_tool", {})


def test_tool_manager_signature_and_definitions():
    tm = ToolManager(timeout=1)
    st = SleepTool(delay=0.01)
    tm.register(st)
    sig = tm.get_signature("sleep_tool")
    assert sig["function"]["name"] == "sleep_tool"
    defs = tm.get_tool_definitions()
    assert any(d["function"]["name"] == "sleep_tool" for d in defs)
