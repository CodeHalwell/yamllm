"""
Performance metrics tracking for YAMLLM.

This module provides performance monitoring capabilities including:
- Request latency tracking
- Token throughput measurement
- Tool execution time monitoring
- Cache hit rate tracking
"""

import time
from typing import Dict, Optional, Any, List
from collections import defaultdict
from dataclasses import dataclass, field
import threading


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    
    # Request metrics
    request_count: int = 0
    total_request_time: float = 0.0
    first_token_latencies: List[float] = field(default_factory=list)
    
    # Token metrics
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    
    # Tool metrics
    tool_execution_count: int = 0
    tool_execution_time: float = 0.0
    tool_execution_times: Dict[str, List[float]] = field(default_factory=dict)
    
    # Cache metrics
    embedding_cache_hits: int = 0
    embedding_cache_misses: int = 0
    tool_def_cache_hits: int = 0
    tool_def_cache_misses: int = 0
    
    def get_avg_request_time(self) -> float:
        """Get average request time in seconds."""
        return self.total_request_time / self.request_count if self.request_count > 0 else 0.0
    
    def get_avg_first_token_latency(self) -> float:
        """Get average first token latency in milliseconds."""
        if not self.first_token_latencies:
            return 0.0
        return sum(self.first_token_latencies) / len(self.first_token_latencies)
    
    def get_p95_first_token_latency(self) -> float:
        """Get 95th percentile first token latency in milliseconds."""
        if not self.first_token_latencies:
            return 0.0
        sorted_latencies = sorted(self.first_token_latencies)
        index = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[index] if index < len(sorted_latencies) else sorted_latencies[-1]
    
    def get_embedding_cache_hit_rate(self) -> float:
        """Get embedding cache hit rate as percentage."""
        total = self.embedding_cache_hits + self.embedding_cache_misses
        return (self.embedding_cache_hits / total * 100) if total > 0 else 0.0
    
    def get_tool_def_cache_hit_rate(self) -> float:
        """Get tool definition cache hit rate as percentage."""
        total = self.tool_def_cache_hits + self.tool_def_cache_misses
        return (self.tool_def_cache_hits / total * 100) if total > 0 else 0.0
    
    def get_tokens_per_second(self, duration: float) -> float:
        """Get token throughput (tokens/second)."""
        return self.total_tokens / duration if duration > 0 else 0.0
    
    def get_avg_tool_execution_time(self, tool_name: Optional[str] = None) -> float:
        """Get average tool execution time in seconds."""
        if tool_name:
            times = self.tool_execution_times.get(tool_name, [])
            return sum(times) / len(times) if times else 0.0
        return self.tool_execution_time / self.tool_execution_count if self.tool_execution_count > 0 else 0.0


class MetricsTracker:
    """
    Thread-safe metrics tracker for performance monitoring.
    
    Tracks various performance metrics including latency, throughput,
    cache hit rates, and tool execution times.
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics = PerformanceMetrics()
        self._lock = threading.Lock()
        self._start_time = time.time()
    
    def record_request_start(self) -> float:
        """Record the start of a request and return timestamp."""
        return time.time()
    
    def record_request_end(self, start_time: float):
        """Record the end of a request."""
        elapsed = time.time() - start_time
        with self._lock:
            self.metrics.request_count += 1
            self.metrics.total_request_time += elapsed
    
    def record_first_token_latency(self, latency_ms: float):
        """Record first token latency in milliseconds."""
        with self._lock:
            self.metrics.first_token_latencies.append(latency_ms)
    
    def record_token_usage(self, prompt_tokens: int, completion_tokens: int):
        """Record token usage."""
        with self._lock:
            self.metrics.total_prompt_tokens += prompt_tokens
            self.metrics.total_completion_tokens += completion_tokens
            self.metrics.total_tokens += prompt_tokens + completion_tokens
    
    def record_tool_execution(self, tool_name: str, execution_time: float):
        """Record tool execution time in seconds."""
        with self._lock:
            self.metrics.tool_execution_count += 1
            self.metrics.tool_execution_time += execution_time
            self.metrics.tool_execution_times[tool_name].append(execution_time)
    
    def record_embedding_cache_hit(self):
        """Record an embedding cache hit."""
        with self._lock:
            self.metrics.embedding_cache_hits += 1
    
    def record_embedding_cache_miss(self):
        """Record an embedding cache miss."""
        with self._lock:
            self.metrics.embedding_cache_misses += 1
    
    def record_tool_def_cache_hit(self):
        """Record a tool definition cache hit."""
        with self._lock:
            self.metrics.tool_def_cache_hits += 1
    
    def record_tool_def_cache_miss(self):
        """Record a tool definition cache miss."""
        with self._lock:
            self.metrics.tool_def_cache_misses += 1
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current metrics snapshot."""
        with self._lock:
            return self.metrics
    
    def get_uptime(self) -> float:
        """Get tracker uptime in seconds."""
        return time.time() - self._start_time
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        with self._lock:
            uptime = self.get_uptime()
            return {
                "uptime_seconds": uptime,
                "requests": {
                    "total": self.metrics.request_count,
                    "avg_time_seconds": self.metrics.get_avg_request_time(),
                },
                "latency": {
                    "avg_first_token_ms": self.metrics.get_avg_first_token_latency(),
                    "p95_first_token_ms": self.metrics.get_p95_first_token_latency(),
                },
                "tokens": {
                    "prompt": self.metrics.total_prompt_tokens,
                    "completion": self.metrics.total_completion_tokens,
                    "total": self.metrics.total_tokens,
                    "throughput_per_second": self.metrics.get_tokens_per_second(uptime),
                },
                "tools": {
                    "execution_count": self.metrics.tool_execution_count,
                    "avg_execution_time_seconds": self.metrics.get_avg_tool_execution_time(),
                },
                "cache": {
                    "embedding_hit_rate_percent": self.metrics.get_embedding_cache_hit_rate(),
                    "tool_def_hit_rate_percent": self.metrics.get_tool_def_cache_hit_rate(),
                },
            }
    
    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self.metrics = PerformanceMetrics()
            self._start_time = time.time()
    
    def _prometheus_requests(self) -> list:
        return [
            "# HELP yamllm_requests_total Total number of requests",
            "# TYPE yamllm_requests_total counter",
            f"yamllm_requests_total {self.metrics.request_count}",
            "",
            "# HELP yamllm_request_duration_seconds Average request duration",
            "# TYPE yamllm_request_duration_seconds gauge",
            f"yamllm_request_duration_seconds {self.metrics.get_avg_request_time():.6f}",
            "",
        ]

    def _prometheus_latency(self) -> list:
        return [
            "# HELP yamllm_first_token_latency_ms Average first token latency",
            "# TYPE yamllm_first_token_latency_ms gauge",
            f"yamllm_first_token_latency_ms {{quantile=\"0.50\"}} {self.metrics.get_avg_first_token_latency():.2f}",
            f"yamllm_first_token_latency_ms {{quantile=\"0.95\"}} {self.metrics.get_p95_first_token_latency():.2f}",
            "",
        ]

    def _prometheus_tokens(self) -> list:
        return [
            "# HELP yamllm_tokens_total Total tokens processed",
            "# TYPE yamllm_tokens_total counter",
            f"yamllm_tokens_total {{type=\"prompt\"}} {self.metrics.total_prompt_tokens}",
            f"yamllm_tokens_total {{type=\"completion\"}} {self.metrics.total_completion_tokens}",
            f"yamllm_tokens_total {{type=\"total\"}} {self.metrics.total_tokens}",
            "",
        ]

    def _prometheus_cache(self) -> list:
        return [
            "# HELP yamllm_cache_hit_rate Cache hit rate percentage",
            "# TYPE yamllm_cache_hit_rate gauge",
            f"yamllm_cache_hit_rate {{cache=\"embedding\"}} {self.metrics.get_embedding_cache_hit_rate():.2f}",
            f"yamllm_cache_hit_rate {{cache=\"tool_def\"}} {self.metrics.get_tool_def_cache_hit_rate():.2f}",
            "",
        ]

    def _prometheus_tools(self) -> list:
        return [
            "# HELP yamllm_tool_executions_total Total tool executions",
            "# TYPE yamllm_tool_executions_total counter",
            f"yamllm_tool_executions_total {self.metrics.tool_execution_count}",
            "",
        ]

    def format_prometheus(self) -> str:
        """Format metrics in Prometheus exposition format."""
        with self._lock:
            lines = []
            lines.extend(self._prometheus_requests())
            lines.extend(self._prometheus_latency())
            lines.extend(self._prometheus_tokens())
            lines.extend(self._prometheus_cache())
            lines.extend(self._prometheus_tools())
            return "\n".join(lines)
