# YAMLLM Performance Guide

This guide covers performance optimization, monitoring, and best practices for YAMLLM.

## Performance Improvements (v0.1.12+)

### 1. Enhanced Caching

#### Embedding Cache (Increased 64 → 1000 entries)
The embedding cache now supports 1000 entries (up from 64) with proper LRU eviction:

```python
# Automatic LRU caching with 1000 entries
llm = LLM(config_path="config.yaml", api_key="...")
embedding = llm.create_embedding("text")  # Cached automatically
```

**Benefits:**
- 15x larger cache capacity
- Proper LRU eviction (least recently used)
- Automatic cache hit tracking
- Expected hit rate: 80-90% for typical workloads

#### Tool Definition Cache
Tool definitions are now cached to avoid regeneration on every request:

```python
# Tool definitions cached based on configuration hash
# Regenerated only when configuration changes
tools = tool_orchestrator.get_tool_definitions()  # First call
tools = tool_orchestrator.get_tool_definitions()  # Cached (instant)
```

**Benefits:**
- Eliminates 5x redundant tool definition generation per request
- Expected hit rate: 95%+ (only misses on config changes)
- Reduces request overhead

### 2. Performance Metrics

Track performance in real-time with built-in metrics:

```python
from yamllm.core.llm import LLM

llm = LLM(config_path="config.yaml", api_key="...")

# Get metrics summary
metrics = llm.get_metrics_summary()
print(f"Avg latency: {metrics['latency']['avg_first_token_ms']}ms")
print(f"Cache hit rate: {metrics['cache']['embedding_hit_rate_percent']}%")

# Export to Prometheus
prometheus_metrics = llm.get_prometheus_metrics()
```

**Tracked Metrics:**
- First token latency (avg, p95)
- Token throughput (tokens/second)
- Cache hit rates (embedding and tool definitions)
- Tool execution times
- Request counts and durations

### 3. Connection Pooling

YAMLLM uses httpx with HTTP/2 support for efficient connection pooling:

```yaml
# config.yaml - already configured
provider:
  name: openai
  # httpx handles connection pooling automatically
```

**Built-in Features:**
- HTTP/2 multiplexing
- Connection reuse across requests
- Automatic retry with backoff
- Configurable timeouts

## Performance Targets

Based on the YAMLLM Manifesto:

| Metric | Target | Status |
|--------|--------|--------|
| First token latency | <350ms | ✓ Optimized |
| Embedding cache hit rate | ≥80% | ✓ Implemented |
| Tool def cache hit rate | ≥95% | ✓ Implemented |
| Metrics overhead | <10µs | ✓ Minimal |

## Monitoring Performance

### 1. Basic Monitoring

```python
# Check performance after operations
summary = llm.get_metrics_summary()

if summary['latency']['avg_first_token_ms'] > 350:
    print("⚠ Latency above target!")

if summary['cache']['embedding_hit_rate_percent'] < 80:
    print("⚠ Cache hit rate below target!")
```

### 2. Prometheus Integration

Export metrics to Prometheus for monitoring:

```python
from flask import Flask, Response
app = Flask(__name__)

@app.route('/metrics')
def metrics():
    return Response(
        llm.get_prometheus_metrics(),
        mimetype='text/plain'
    )
```

### 3. Continuous Benchmarking

Run benchmarks regularly:

```bash
# Run performance tests
pytest tests/performance/

# Run benchmark suite
python tests/performance/benchmark_runner.py
```

## Token Usage Optimization

### 1. Reduce Token Usage with Tool Filtering

YAMLLM automatically filters tools based on intent:

```yaml
# config.yaml
tools:
  enabled: true
  gate_web_search: true  # Enable intelligent tool filtering
```

**How it works:**
- Analyzes prompt intent (web search, calculation, etc.)
- Only includes relevant tools in API call
- Reduces token usage by excluding irrelevant tools
- Example: Only includes calculator for math queries

### 2. Optimize System Prompts

```yaml
# config.yaml
context:
  system_prompt: |
    Be concise and helpful.
  # Shorter prompts = fewer tokens
```

**Tips:**
- Remove unnecessary instructions
- Use clear, concise language
- Avoid repetitive context
- Leverage tool descriptions instead of examples

### 3. Use Streaming for Better UX

```python
# Streaming reduces perceived latency
response = llm.chat("Hello", stream=True)
# User sees output immediately
```

### 4. Memory Management

```yaml
# config.yaml
memory:
  enabled: true
  max_history: 5  # Limit conversation history
  # Fewer messages = fewer tokens
```

### 5. Monitor Token Usage

```python
# Track token usage
summary = llm.get_metrics_summary()
print(f"Total tokens: {summary['tokens']['total']}")
print(f"Avg per request: {summary['tokens']['total'] / summary['requests']['total']}")
```

## Best Practices

### 1. Cache Optimization

```python
# Pre-warm cache with common queries
common_texts = ["user info", "help text", "documentation"]
for text in common_texts:
    llm.create_embedding(text)  # Cache for later use
```

### 2. Batch Operations

```python
# Batch embeddings when possible
texts = ["text1", "text2", "text3"]
embeddings = [llm.create_embedding(t) for t in texts]
# Cache hit rate improves with repeated patterns
```

### 3. Tool Selection

```yaml
# Only enable needed tools
tools:
  enabled: true
  tool_list: [calculator, web_search]  # Specific tools
  # OR
  tool_packs: [common, web]  # Tool packs
```

### 4. Request Optimization

```python
# Use provider-specific optimizations
llm = LLM(config_path="config.yaml", api_key="...")

# Set appropriate timeouts
llm.update_settings(timeout=30)

# Use reasonable token limits
llm.update_settings(max_tokens=1000)
```

## Troubleshooting Performance Issues

### High Latency

**Symptoms:** First token latency >350ms consistently

**Solutions:**
1. Check network connectivity
2. Verify API endpoint is optimal (region)
3. Review system prompt length
4. Check tool count (filter tools)
5. Monitor provider status

### Low Cache Hit Rate

**Symptoms:** Cache hit rate <80%

**Solutions:**
1. Review access patterns (are queries unique?)
2. Increase cache size if needed
3. Pre-warm cache with common queries
4. Check for unnecessary cache invalidation

### High Token Usage

**Symptoms:** Excessive token consumption

**Solutions:**
1. Enable tool filtering (`gate_web_search: true`)
2. Reduce system prompt length
3. Limit conversation history
4. Use streaming for better UX
5. Review tool descriptions (shorter is better)

### Memory Issues

**Symptoms:** High memory usage

**Solutions:**
1. Limit cache size if necessary
2. Clear old conversation history
3. Use vector store with appropriate settings
4. Monitor metrics memory footprint

## Performance Testing

### Run Tests

```bash
# All performance tests
pytest tests/performance/ -v

# Specific test
pytest tests/performance/test_latency.py -v

# With coverage
pytest tests/performance/ --cov=yamllm --cov-report=html
```

### Benchmark Suite

```bash
# Run benchmarks
python tests/performance/benchmark_runner.py

# Output:
# YAMLLM PERFORMANCE BENCHMARK REPORT
# ====================================
# Overall Status: ✓ ALL TESTS PASSED
# 
# Embedding Cache: 90.0% hit rate ✓
# Tool Def Cache: 99.0% hit rate ✓
# Metrics Overhead: 2.5µs ✓
```

## Example: Complete Performance Monitoring

```python
from yamllm.core.llm import LLM
import time

# Initialize with monitoring
llm = LLM(config_path="config.yaml", api_key="...")

# Perform operations
start = time.time()
response = llm.chat("What is 2+2?")
duration = time.time() - start

# Check performance
metrics = llm.get_metrics_summary()

print(f"Response time: {duration:.3f}s")
print(f"First token latency: {metrics['latency']['avg_first_token_ms']:.2f}ms")
print(f"Tokens used: {metrics['tokens']['total']}")
print(f"Cache hit rate: {metrics['cache']['embedding_hit_rate_percent']:.1f}%")

# Alert on performance issues
if metrics['latency']['avg_first_token_ms'] > 350:
    print("⚠ Performance degradation detected!")
    
# Export for monitoring
with open('/var/metrics/yamllm.prom', 'w') as f:
    f.write(llm.get_prometheus_metrics())
```

## References

- [YAMLLM Manifesto](../yamllm_manifesto.md) - Performance targets
- [Performance Tests](../tests/performance/) - Test suite
- [Benchmark Runner](../tests/performance/benchmark_runner.py) - Benchmarking tool
- [Metrics Example](../examples/metrics_example.py) - Example usage
