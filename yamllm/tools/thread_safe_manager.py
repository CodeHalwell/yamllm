"""
Thread-safe tool manager implementation.

This module provides a thread-safe version of the tool manager that prevents
race conditions when multiple tools access shared resources.
"""

import threading
import concurrent.futures
import time
from typing import Dict, Any, List, Optional, Set
from contextlib import contextmanager

from yamllm.tools.manager import ToolExecutor
import logging
from yamllm.core.exceptions import ToolExecutionError


class ThreadSafeToolManager(ToolExecutor):
    """
    Thread-safe tool manager with resource locking and execution tracking.
    
    This class extends ToolExecutor with thread safety features to prevent
    race conditions when tools access shared resources.
    """
    
    def __init__(self, *, timeout: int = 30, max_concurrent: int = 5, logger=None):
        """
        Initialize thread-safe tool manager.
        
        Args:
            timeout: Timeout for individual tool execution
            max_concurrent: Maximum concurrent tool executions
            logger: Logger instance
        """
        super().__init__(timeout=timeout, logger=logger or logging.getLogger("yamllm.tools"))
        
        # Thread safety primitives
        self._lock = threading.RLock()
        self._resource_locks: Dict[str, threading.Lock] = {}
        self._execution_semaphore = threading.Semaphore(max_concurrent)
        self._max_concurrent = max_concurrent
        
        # Track active executions
        self._active_executions: Set[str] = set()
        self._execution_count = 0
        self._execution_lock = threading.Lock()
        
        # Thread pool for concurrent execution
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_concurrent,
            thread_name_prefix="tool_executor"
        )
    
    def register(self, tool) -> None:
        """Thread-safe tool registration."""
        with self._lock:
            super().register(tool)
            
            # Create resource lock for tools that need exclusive access
            if hasattr(tool, 'requires_exclusive_access') and tool.requires_exclusive_access:
                self._resource_locks[tool.name] = threading.Lock()
    
    def execute(self, name: str, args: Dict[str, Any]) -> Any:
        """
        Thread-safe tool execution with resource locking.
        
        Args:
            name: Tool name
            args: Tool arguments
            
        Returns:
            Tool execution result
        """
        # Acquire execution slot
        if not self._execution_semaphore.acquire(timeout=self._timeout):
            raise ToolExecutionError(
                tool_name=name,
                message="Tool execution queue is full",
                tool_args=args
            )
        
        try:
            # Track execution
            with self._execution_lock:
                if name in self._active_executions:
                    # Tool is already executing - potential recursive call
                    msg = f"Tool '{name}' is already executing - possible recursive call"
                    self._logger.warning(msg)
                    logging.getLogger().warning(msg)
                self._active_executions.add(name)
                self._execution_count += 1
                execution_id = self._execution_count
            
            self._logger.debug(f"Starting execution {execution_id} for tool '{name}'")
            
            # Get resource lock if needed
            resource_lock = self._resource_locks.get(name)
            
            if resource_lock:
                self._logger.debug(f"Acquiring resource lock for tool '{name}'")
                lock_acquired = resource_lock.acquire(timeout=self._timeout)
                if not lock_acquired:
                    raise ToolExecutionError(
                        tool_name=name,
                        message="Failed to acquire resource lock",
                        tool_args=args
                    )
            
            try:
                # Execute tool
                start_time = time.time()
                result = self._execute_tool_safely(name, args)
                execution_time = time.time() - start_time
                
                self._logger.debug(
                    f"Completed execution {execution_id} for tool '{name}' "
                    f"in {execution_time:.2f}s"
                )
                
                return result
                
            finally:
                # Release resource lock
                if resource_lock:
                    resource_lock.release()
                    self._logger.debug(f"Released resource lock for tool '{name}'")
            
        finally:
            # Clean up execution tracking
            with self._execution_lock:
                self._active_executions.discard(name)
            
            # Release execution slot
            self._execution_semaphore.release()
    
    def _execute_tool_safely(self, name: str, args: Dict[str, Any]) -> Any:
        """Execute tool with proper error handling."""
        tool = self._require(name)
        
        # Submit to thread pool
        future = self._executor.submit(tool.execute, **args)
        
        try:
            return future.result(timeout=self._timeout)
        except concurrent.futures.TimeoutError:
            # Cancel the future if possible
            future.cancel()
            raise ToolExecutionError(
                tool_name=name,
                message=f"Timed out after {self._timeout}s",
                tool_args=args,
            )
        except Exception as e:
            raise ToolExecutionError(
                tool_name=name,
                message=str(e),
                original_error=e,
                tool_args=args,
            )
    
    def execute_many(
        self, executions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple tools concurrently with proper synchronization.
        
        Args:
            executions: List of dicts with 'name' and 'args' keys
            
        Returns:
            List of results in the same order as inputs
        """
        results: List[Dict[str, Any]] = []
        # Use a separate executor to avoid nested deadlock with self._executor
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_concurrent) as outer:
            futures: List[tuple[concurrent.futures.Future, str]] = []
            for execution in executions:
                name = execution['name']
                args = execution.get('args', {})
                fut = outer.submit(self.execute, name, args)
                futures.append((fut, name))

            for future, name in futures:
                try:
                    result = future.result(timeout=self._timeout * 2)
                    results.append({'success': True, 'result': result, 'tool': name})
                except Exception as e:
                    results.append({'success': False, 'error': str(e), 'tool': name})

        return results
    
    @contextmanager
    def acquire_resource(self, resource_name: str, timeout: Optional[float] = None):
        """
        Context manager for acquiring a named resource lock.
        
        Args:
            resource_name: Name of the resource to lock
            timeout: Lock acquisition timeout
            
        Yields:
            None when lock is acquired
            
        Raises:
            ToolExecutionError: If lock cannot be acquired
        """
        # Get or create lock for resource
        with self._lock:
            if resource_name not in self._resource_locks:
                self._resource_locks[resource_name] = threading.Lock()
            lock = self._resource_locks[resource_name]
        
        # Acquire lock
        timeout = timeout or self._timeout
        acquired = lock.acquire(timeout=timeout)
        
        if not acquired:
            raise ToolExecutionError(
                tool_name="resource_manager",
                message=f"Failed to acquire lock for resource '{resource_name}'"
            )
        
        try:
            yield
        finally:
            lock.release()
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        with self._execution_lock:
            return {
                "total_executions": self._execution_count,
                "active_executions": list(self._active_executions),
                "active_count": len(self._active_executions),
                "max_concurrent": self._max_concurrent
            }
    
    def close(self):
        """Clean up resources."""
        self._executor.shutdown(wait=True)
        super().close()

    def execute_safe(self, name: str, args: Dict[str, Any]) -> Any:
        """Execute and return {'error': str} on failures instead of raising."""
        try:
            return self.execute(name, args)
        except ToolExecutionError as e:
            return {"error": str(e)}


class ResourceAwareTool:
    """
    Base class for tools that need exclusive resource access.
    
    Tools that extend this class will automatically get exclusive
    execution when registered with ThreadSafeToolManager.
    """
    
    requires_exclusive_access = True
    
    def __init__(self, resource_name: str = None):
        """
        Initialize resource-aware tool.
        
        Args:
            resource_name: Name of the resource this tool accesses
        """
        self.resource_name = resource_name or self.__class__.__name__
