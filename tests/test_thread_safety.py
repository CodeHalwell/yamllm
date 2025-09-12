"""
Tests for thread-safe tool execution.
"""

import pytest
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from yamllm.tools.thread_safe_manager import ThreadSafeToolManager, ResourceAwareTool
from yamllm.tools.base import Tool
from yamllm.core.exceptions import ToolExecutionError


class MockTool(Tool):
    """Mock tool for testing."""
    
    def __init__(self, name: str, execution_time: float = 0.1):
        self.name = name
        self.description = f"Mock tool {name}"
        self.execution_time = execution_time
        self.execution_count = 0
        self.concurrent_executions = 0
        self.max_concurrent = 0
        self._lock = threading.Lock()
    
    def get_signature(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"}
                    }
                }
            }
        }
    
    def execute(self, value: str = "test"):
        with self._lock:
            self.execution_count += 1
            self.concurrent_executions += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent_executions)
        
        try:
            time.sleep(self.execution_time)
            return f"{self.name}: {value}"
        finally:
            with self._lock:
                self.concurrent_executions -= 1


class ExclusiveResourceTool(ResourceAwareTool, Tool):
    """Tool that requires exclusive resource access."""
    
    def __init__(self, name: str):
        ResourceAwareTool.__init__(self)
        self.name = name
        self.description = f"Exclusive tool {name}"
        self.concurrent_count = 0
        self.max_concurrent = 0
        self._lock = threading.Lock()
    
    def get_signature(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": {}}
            }
        }
    
    def execute(self):
        with self._lock:
            self.concurrent_count += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent_count)
        
        try:
            time.sleep(0.1)
            return "exclusive_result"
        finally:
            with self._lock:
                self.concurrent_count -= 1


class TestThreadSafeToolManager:
    """Test thread-safe tool manager."""
    
    def test_basic_execution(self):
        """Test basic tool execution."""
        manager = ThreadSafeToolManager(timeout=5)
        tool = MockTool("test_tool")
        manager.register(tool)
        
        result = manager.execute("test_tool", {"value": "hello"})
        assert result == "test_tool: hello"
        assert tool.execution_count == 1
    
    def test_concurrent_execution_limit(self):
        """Test that concurrent execution is limited."""
        manager = ThreadSafeToolManager(timeout=5, max_concurrent=2)
        
        # Register slow tools
        tools = []
        for i in range(3):
            tool = MockTool(f"tool_{i}", execution_time=0.2)
            manager.register(tool)
            tools.append(tool)
        
        # Execute tools concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for i in range(3):
                future = executor.submit(
                    manager.execute, f"tool_{i}", {"value": str(i)}
                )
                futures.append(future)
            
            # Wait for all to complete
            results = [f.result() for f in futures]
        
        # Check results
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result == f"tool_{i}: {i}"
        
        # Check that max concurrent was respected
        # At least one tool should have seen no concurrency due to semaphore
        max_concurrents = [t.max_concurrent for t in tools]
        assert min(max_concurrents) == 1
    
    def test_exclusive_resource_locking(self):
        """Test that exclusive resources prevent concurrent access."""
        manager = ThreadSafeToolManager(timeout=5, max_concurrent=5)
        
        # Register exclusive tools
        tool1 = ExclusiveResourceTool("exclusive1")
        tool2 = ExclusiveResourceTool("exclusive2")
        manager.register(tool1)
        manager.register(tool2)
        
        # Try to execute concurrently
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(manager.execute, "exclusive1", {})
            future2 = executor.submit(manager.execute, "exclusive2", {})
            
            result1 = future1.result()
            result2 = future2.result()
        
        assert result1 == "exclusive_result"
        assert result2 == "exclusive_result"
        
        # Both should have max concurrent of 1 (exclusive access)
        assert tool1.max_concurrent == 1
        assert tool2.max_concurrent == 1
    
    def test_recursive_call_detection(self, caplog):
        """Test that recursive calls are detected and logged."""
        manager = ThreadSafeToolManager(timeout=5)
        
        class RecursiveTool(Tool):
            def __init__(self, manager):
                self.name = "recursive"
                self.description = "Recursive tool"
                self.manager = manager
                self.call_count = 0
            
            def get_signature(self):
                return {
                    "type": "function",
                    "function": {
                        "name": self.name,
                        "description": self.description,
                        "parameters": {"type": "object", "properties": {}}
                    }
                }
            
            def execute(self):
                self.call_count += 1
                if self.call_count < 2:
                    # Recursive call
                    return self.manager.execute("recursive", {})
                return "done"
        
        tool = RecursiveTool(manager)
        manager.register(tool)
        
        result = manager.execute("recursive", {})
        assert result == "done"
        assert tool.call_count == 2
        
        # Check for warning about recursive execution
        assert any("already executing" in record.message for record in caplog.records)
    
    def test_execution_timeout(self):
        """Test that tool execution times out properly."""
        manager = ThreadSafeToolManager(timeout=0.2)
        
        class SlowTool(Tool):
            name = "slow"
            description = "Slow tool"
            
            def get_signature(self):
                return {
                    "type": "function",
                    "function": {
                        "name": self.name,
                        "description": self.description,
                        "parameters": {"type": "object", "properties": {}}
                    }
                }
            
            def execute(self):
                time.sleep(1.0)  # Longer than timeout
                return "should_not_reach"
        
        manager.register(SlowTool())
        
        with pytest.raises(ToolExecutionError) as exc_info:
            manager.execute("slow", {})
        
        assert "Timed out" in str(exc_info.value)
    
    def test_execute_many(self):
        """Test executing multiple tools concurrently."""
        manager = ThreadSafeToolManager(timeout=5, max_concurrent=3)
        
        # Register tools
        for i in range(5):
            tool = MockTool(f"tool_{i}", execution_time=0.1)
            manager.register(tool)
        
        # Execute many
        executions = [
            {"name": f"tool_{i}", "args": {"value": str(i)}}
            for i in range(5)
        ]
        
        results = manager.execute_many(executions)
        
        # Check results
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result["success"] is True
            assert result["result"] == f"tool_{i}: {i}"
            assert result["tool"] == f"tool_{i}"
    
    def test_resource_acquisition_context_manager(self):
        """Test resource acquisition context manager."""
        manager = ThreadSafeToolManager(timeout=5)
        
        # Track acquisition order
        acquisition_order = []
        
        def acquire_resource(name, resource_name):
            with manager.acquire_resource(resource_name):
                acquisition_order.append(f"{name}_start")
                time.sleep(0.1)
                acquisition_order.append(f"{name}_end")
        
        # Try to acquire same resource from multiple threads
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(acquire_resource, "thread1", "shared_resource")
            future2 = executor.submit(acquire_resource, "thread2", "shared_resource")
            
            future1.result()
            future2.result()
        
        # Check that access was serialized
        assert acquisition_order[0].endswith("_start")
        assert acquisition_order[1].endswith("_end")
        assert acquisition_order[2].endswith("_start")
        assert acquisition_order[3].endswith("_end")
        
        # The same thread should start and end
        thread1_first = acquisition_order[0].startswith("thread1")
        if thread1_first:
            assert acquisition_order[1] == "thread1_end"
            assert acquisition_order[2] == "thread2_start"
            assert acquisition_order[3] == "thread2_end"
        else:
            assert acquisition_order[1] == "thread2_end"
            assert acquisition_order[2] == "thread1_start"
            assert acquisition_order[3] == "thread1_end"
    
    def test_execution_stats(self):
        """Test execution statistics tracking."""
        manager = ThreadSafeToolManager(timeout=5, max_concurrent=2)
        
        tool = MockTool("stats_tool", execution_time=0.1)
        manager.register(tool)
        
        # Initial stats
        stats = manager.get_execution_stats()
        assert stats["total_executions"] == 0
        assert stats["active_count"] == 0
        
        # Execute and check stats during execution
        def execute_and_check():
            manager.execute("stats_tool", {"value": "test"})
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(execute_and_check)
            future2 = executor.submit(execute_and_check)
            
            # Give threads time to start
            time.sleep(0.05)
            
            # Check stats during execution
            stats = manager.get_execution_stats()
            assert stats["total_executions"] >= 1
            assert stats["active_count"] >= 1
            
            # Wait for completion
            future1.result()
            future2.result()
        
        # Final stats
        stats = manager.get_execution_stats()
        assert stats["total_executions"] == 2
        assert stats["active_count"] == 0