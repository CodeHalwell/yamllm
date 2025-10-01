"""
Async tool manager for YAMLLM.

This module provides async execution capabilities for tools,
enabling concurrent tool execution and better performance.
"""

import asyncio
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

from yamllm.tools.manager import ToolExecutor
from yamllm.core.exceptions import ToolExecutionError


class AsyncToolManager(ToolExecutor):
    """
    Async-enabled tool manager.
    
    This class extends ToolExecutor with async execution capabilities,
    allowing tools to be run concurrently.
    """
    
    def __init__(self, *, timeout: int = 30, max_workers: int = 5, logger=None):
        """
        Initialize async tool manager.
        
        Args:
            timeout: Timeout for tool execution
            max_workers: Maximum concurrent tool executions
            logger: Logger instance
        """
        super().__init__(timeout=timeout, logger=logger)
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def execute_async(self, name: str, args: Dict[str, Any]) -> Any:
        """
        Execute a tool asynchronously.
        
        Args:
            name: Tool name
            args: Tool arguments
            
        Returns:
            Tool execution result
            
        Raises:
            ToolExecutionError: If tool execution fails
        """
        tool = self._require(name)
        
        try:
            # Run the tool in a thread pool since most tools are sync
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(self._executor, tool.execute, **args),
                timeout=self._timeout
            )
            return result
            
        except asyncio.TimeoutError:
            error_msg = f"Tool '{name}' timed out after {self._timeout}s"
            if self._logger:
                self._logger.warning(error_msg)
            raise ToolExecutionError(name, error_msg)
            
        except Exception as e:
            error_msg = f"Tool execution failed: {e}"
            if self._logger:
                self._logger.error(f"Tool '{name}': {error_msg}")
            raise ToolExecutionError(name, error_msg, e)
    
    async def execute_many_async(
        self, executions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple tools concurrently.
        
        Args:
            executions: List of dicts with 'name' and 'args' keys
            
        Returns:
            List of results in the same order as inputs
        """
        tasks = []
        for execution in executions:
            name = execution['name']
            args = execution.get('args', {})
            task = self.execute_async(name, args)
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'success': False,
                    'error': str(result),
                    'tool': executions[i]['name']
                })
            else:
                processed_results.append({
                    'success': True,
                    'result': result,
                    'tool': executions[i]['name']
                })
        
        return processed_results
    
    def close(self):
        """Clean up resources."""
        self._executor.shutdown(wait=True)
    
    async def aclose(self):
        """Async cleanup."""
        await asyncio.get_event_loop().run_in_executor(
            None, self._executor.shutdown, True
        )