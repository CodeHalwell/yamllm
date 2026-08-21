# YAMLLM Performance Tests

This directory contains performance tests and benchmarks for YAMLLM.

## Overview

The performance test suite validates that YAMLLM meets its performance targets as outlined in the manifesto:
- First token latency < 350ms
- Efficient caching (>80% hit rate)
- Minimal overhead from metrics tracking

## Running Tests

### Run all performance tests
```bash
pytest tests/performance/
```

### Run specific test file
```bash
pytest tests/performance/test_latency.py
```

### Run benchmarks
```bash
python tests/performance/benchmark_runner.py
```

## Test Files

### `test_latency.py`
Tests for latency tracking, throughput measurement, and cache performance:
- `TestFirstTokenLatency` - First token latency tracking and targets
- `TestThroughput` - Token throughput calculations
- `TestCachePerformance` - Cache hit rates and effectiveness
- `TestMemoryUsage` - Memory usage baselines
- `TestToolExecutionTime` - Tool execution time tracking
- `TestMetricsSummary` - Metrics reporting and Prometheus format

### `benchmark_runner.py`
Standalone benchmark runner that:
- Measures embedding cache performance
- Measures tool definition cache performance
- Measures metrics tracking overhead
- Generates performance report

## Performance Targets

### Latency
- **First token**: <350ms (avg)
- **P95 latency**: <500ms

### Throughput
- **Tokens/second**: Monitor and track
- **Requests/minute**: Monitor and track

### Caching
- **Embedding cache**: ≥80% hit rate
- **Tool definition cache**: ≥95% hit rate
- **Cache size**: 1000 entries (increased from 64)

### Overhead
- **Metrics tracking**: <10µs per operation

## Metrics Available

The metrics system tracks:
- Request latency (avg, p95)
- Token usage (prompt, completion, total)
- Token throughput
- Tool execution times (overall and per-tool)
- Cache hit rates (embedding and tool definition)

### Accessing Metrics

```python
from yamllm.core.llm import LLM

llm = LLM(config_path="config.yaml", api_key="...")

# Get metrics summary
summary = llm.get_metrics_summary()
print(summary)

# Get Prometheus format
prometheus = llm.get_prometheus_metrics()
print(prometheus)

# Reset metrics
llm.reset_metrics()
```

## Continuous Monitoring

For production deployments, consider:
1. Export metrics to Prometheus/Grafana
2. Set up alerts for performance degradation
3. Run benchmarks periodically to detect regressions
4. Monitor cache hit rates to optimize cache size

## Adding New Tests

When adding new performance tests:
1. Follow the existing test structure
2. Use `MetricsTracker` for measurements
3. Set clear performance targets
4. Document expected behavior
5. Add to the benchmark runner if appropriate
