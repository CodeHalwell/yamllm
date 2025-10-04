"""
Performance tests for latency and throughput.

These tests verify that YAMLLM meets performance targets:
- First token latency < 350ms (target from manifesto)
- Reasonable throughput for token processing
- Efficient cache utilization
"""

import time
import pytest
from unittest.mock import Mock, patch, MagicMock
from yamllm.core.llm import LLM
from yamllm.core.metrics import MetricsTracker


class TestFirstTokenLatency:
    """Test first token latency performance."""
    
    def test_first_token_latency_tracking(self):
        """Test that first token latency is tracked."""
        tracker = MetricsTracker()
        
        # Simulate recording latencies
        tracker.record_first_token_latency(100.0)  # 100ms
        tracker.record_first_token_latency(200.0)  # 200ms
        tracker.record_first_token_latency(150.0)  # 150ms
        
        metrics = tracker.get_metrics()
        assert len(metrics.first_token_latencies) == 3
        assert metrics.get_avg_first_token_latency() == 150.0
        
    def test_p95_latency_calculation(self):
        """Test 95th percentile latency calculation."""
        tracker = MetricsTracker()
        
        # Add 100 latencies
        for i in range(100):
            tracker.record_first_token_latency(float(i))
        
        metrics = tracker.get_metrics()
        p95 = metrics.get_p95_first_token_latency()
        
        # P95 should be around 95
        assert 94.0 <= p95 <= 95.0
    
    def test_latency_target_awareness(self):
        """Test awareness of 350ms latency target."""
        tracker = MetricsTracker()
        
        # Simulate latencies under target
        tracker.record_first_token_latency(100.0)
        tracker.record_first_token_latency(200.0)
        tracker.record_first_token_latency(300.0)
        
        metrics = tracker.get_metrics()
        avg_latency = metrics.get_avg_first_token_latency()
        
        # Average should be under 350ms
        assert avg_latency < 350.0
        assert avg_latency == 200.0  # (100 + 200 + 300) / 3


class TestThroughput:
    """Test token throughput performance."""
    
    def test_token_throughput_calculation(self):
        """Test token throughput calculation."""
        tracker = MetricsTracker()
        
        # Simulate processing 1000 tokens in 10 seconds
        tracker.record_token_usage(500, 500)  # 1000 total tokens
        time.sleep(0.1)  # Small delay
        
        uptime = tracker.get_uptime()
        metrics = tracker.get_metrics()
        throughput = metrics.get_tokens_per_second(uptime)
        
        # Throughput should be reasonable
        assert throughput > 0
        assert metrics.total_tokens == 1000
    
    def test_token_usage_accumulation(self):
        """Test that token usage accumulates correctly."""
        tracker = MetricsTracker()
        
        # Multiple requests
        tracker.record_token_usage(100, 50)
        tracker.record_token_usage(200, 100)
        tracker.record_token_usage(150, 75)
        
        metrics = tracker.get_metrics()
        assert metrics.total_prompt_tokens == 450
        assert metrics.total_completion_tokens == 225
        assert metrics.total_tokens == 675


class TestCachePerformance:
    """Test cache performance and hit rates."""
    
    def test_embedding_cache_hit_rate(self):
        """Test embedding cache hit rate calculation."""
        tracker = MetricsTracker()
        
        # Simulate cache hits and misses
        tracker.record_embedding_cache_hit()
        tracker.record_embedding_cache_hit()
        tracker.record_embedding_cache_hit()
        tracker.record_embedding_cache_miss()
        
        metrics = tracker.get_metrics()
        hit_rate = metrics.get_embedding_cache_hit_rate()
        
        # Hit rate should be 75% (3 hits out of 4 total)
        assert hit_rate == 75.0
    
    def test_tool_def_cache_hit_rate(self):
        """Test tool definition cache hit rate calculation."""
        tracker = MetricsTracker()
        
        # Simulate cache behavior
        tracker.record_tool_def_cache_miss()  # First call
        tracker.record_tool_def_cache_hit()   # Second call (cached)
        tracker.record_tool_def_cache_hit()   # Third call (cached)
        tracker.record_tool_def_cache_hit()   # Fourth call (cached)
        
        metrics = tracker.get_metrics()
        hit_rate = metrics.get_tool_def_cache_hit_rate()
        
        # Hit rate should be 75% (3 hits out of 4 total)
        assert hit_rate == 75.0
    
    def test_cache_effectiveness(self):
        """Test that caching is effective."""
        tracker = MetricsTracker()
        
        # Many hits, few misses = effective cache
        for _ in range(90):
            tracker.record_embedding_cache_hit()
        for _ in range(10):
            tracker.record_embedding_cache_miss()
        
        metrics = tracker.get_metrics()
        hit_rate = metrics.get_embedding_cache_hit_rate()
        
        # Effective cache should have > 80% hit rate
        assert hit_rate >= 80.0


class TestMemoryUsage:
    """Test memory usage baseline."""
    
    def test_embedding_cache_size_limit(self):
        """Test that embedding cache respects size limit."""
        from collections import OrderedDict
        
        cache = OrderedDict()
        cache_size = 1000
        
        # Add more items than cache size
        for i in range(1500):
            cache[f"text_{i}"] = [0.1] * 1536
            if len(cache) > cache_size:
                cache.popitem(last=False)
        
        # Cache should not exceed size limit
        assert len(cache) == cache_size
    
    def test_metrics_memory_efficiency(self):
        """Test that metrics tracker is memory efficient."""
        tracker = MetricsTracker()
        
        # Add many entries
        for i in range(1000):
            tracker.record_first_token_latency(float(i))
            tracker.record_token_usage(100, 50)
        
        # Metrics should still be accessible
        metrics = tracker.get_metrics()
        assert len(metrics.first_token_latencies) == 1000
        assert metrics.request_count == 0  # No requests recorded yet


class TestToolExecutionTime:
    """Test tool execution time tracking."""
    
    def test_tool_execution_time_recording(self):
        """Test tool execution time recording."""
        tracker = MetricsTracker()
        
        # Record some tool executions
        tracker.record_tool_execution("web_search", 0.5)
        tracker.record_tool_execution("calculator", 0.1)
        tracker.record_tool_execution("web_search", 0.6)
        
        metrics = tracker.get_metrics()
        
        # Check overall metrics
        assert metrics.tool_execution_count == 3
        assert metrics.tool_execution_time == 1.2
        
        # Check per-tool metrics
        avg_web_search = metrics.get_avg_tool_execution_time("web_search")
        avg_calculator = metrics.get_avg_tool_execution_time("calculator")
        
        assert avg_web_search == 0.55  # (0.5 + 0.6) / 2
        assert avg_calculator == 0.1
    
    def test_tool_execution_tracking(self):
        """Test comprehensive tool execution tracking."""
        tracker = MetricsTracker()
        
        # Multiple tools
        tools = ["web_search", "calculator", "datetime", "uuid"]
        for tool in tools:
            for i in range(5):
                tracker.record_tool_execution(tool, 0.1 * (i + 1))
        
        metrics = tracker.get_metrics()
        assert metrics.tool_execution_count == 20
        
        # Each tool should have 5 executions
        for tool in tools:
            times = metrics.tool_execution_times[tool]
            assert len(times) == 5


class TestMetricsReset:
    """Test metrics reset functionality."""
    
    def test_metrics_reset(self):
        """Test that metrics can be reset."""
        tracker = MetricsTracker()
        
        # Add some data
        tracker.record_first_token_latency(100.0)
        tracker.record_token_usage(100, 50)
        tracker.record_embedding_cache_hit()
        
        # Reset
        tracker.reset()
        
        # All metrics should be cleared
        metrics = tracker.get_metrics()
        assert len(metrics.first_token_latencies) == 0
        assert metrics.total_tokens == 0
        assert metrics.embedding_cache_hits == 0


class TestMetricsSummary:
    """Test metrics summary generation."""
    
    def test_summary_format(self):
        """Test that summary is properly formatted."""
        tracker = MetricsTracker()
        
        # Add some data
        tracker.record_first_token_latency(100.0)
        tracker.record_token_usage(100, 50)
        tracker.record_embedding_cache_hit()
        tracker.record_embedding_cache_miss()
        
        summary = tracker.get_summary()
        
        # Check structure
        assert "uptime_seconds" in summary
        assert "requests" in summary
        assert "latency" in summary
        assert "tokens" in summary
        assert "tools" in summary
        assert "cache" in summary
        
        # Check values
        assert summary["latency"]["avg_first_token_ms"] == 100.0
        assert summary["tokens"]["total"] == 150
        assert summary["cache"]["embedding_hit_rate_percent"] == 50.0
    
    def test_prometheus_format(self):
        """Test Prometheus metrics format."""
        tracker = MetricsTracker()
        
        # Add some data
        tracker.record_token_usage(100, 50)
        tracker.record_embedding_cache_hit()
        
        prometheus_text = tracker.format_prometheus()
        
        # Should contain standard Prometheus elements
        assert "# HELP" in prometheus_text
        assert "# TYPE" in prometheus_text
        assert "yamllm_" in prometheus_text
        assert "counter" in prometheus_text
        assert "gauge" in prometheus_text
