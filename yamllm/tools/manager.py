from __future__ import annotations

import concurrent.futures
import time
from typing import Dict, Any, List

from .base import Tool
from yamllm.core.exceptions import ToolExecutionError


class ToolManager:
    """
    Central registry and executor for tools.

    - Registers tool instances
    - Exposes provider-friendly tool definitions (function tools schema)
    - Executes tools with timeouts and normalized errors
    """

    def __init__(self, *, timeout: int = 30, logger=None) -> None:
        self._tools: Dict[str, Tool] = {}
        self._timeout = timeout
        self._logger = logger

    # Registration
    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def register_many(self, tools: List[Tool]) -> None:
        for t in tools:
            self.register(t)

    # Introspection
    def list(self) -> List[str]:
        return list(self._tools.keys())

    def get_signature(self, name: str) -> Dict[str, Any]:
        tool = self._require(name)
        return tool.get_signature()

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        return [t.get_signature() for t in self._tools.values()]

    # Execution
    def execute(self, name: str, args: Dict[str, Any]) -> Any:
        tool = self._require(name)
        start_time = time.time()
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(tool.execute, **args)
                return fut.result(timeout=self._timeout)
                
        except concurrent.futures.TimeoutError:
            execution_time = time.time() - start_time
            error = ToolExecutionError(
                tool_name=name,
                message=f"timed out after {self._timeout}s",
                tool_args=args,
                execution_time=execution_time
            )
            error.log_error()
            raise error
            
        except KeyError as e:
            execution_time = time.time() - start_time
            error = ToolExecutionError(
                tool_name=name,
                message=f"Missing required argument: {e}",
                original_error=e,
                tool_args=args,
                execution_time=execution_time
            )
            error.log_error()
            raise error
            
        except (ValueError, TypeError) as e:
            execution_time = time.time() - start_time
            error = ToolExecutionError(
                tool_name=name,
                message=f"Invalid argument type or value: {e}",
                original_error=e,
                tool_args=args,
                execution_time=execution_time
            )
            error.log_error()
            raise error
            
        except ToolExecutionError:
            # Re-raise tool execution errors
            raise
            
        except Exception as e:
            execution_time = time.time() - start_time
            error = ToolExecutionError(
                tool_name=name,
                message=f"Unexpected error: {e}",
                original_error=e,
                tool_args=args,
                execution_time=execution_time
            )
            error.log_error()
            raise error

    def execute_safe(self, name: str, args: Dict[str, Any]) -> Any:
        """Execute and return {'error': str} on failures instead of raising.

        Useful for UI call-sites that prefer soft-failures.
        """
        try:
            return self.execute(name, args)
        except ToolExecutionError as e:
            return {"error": str(e)}

    # Internal helpers
    def _require(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not registered")
        return self._tools[name]
