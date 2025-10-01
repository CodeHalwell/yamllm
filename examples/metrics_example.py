"""
Example demonstrating performance metrics tracking in YAMLLM.

This example shows how to:
1. Access performance metrics
2. Monitor cache hit rates
3. Export metrics to Prometheus format
"""

import os
from yamllm.core.llm import LLM


def main():
    """Demonstrate metrics tracking."""
    
    # Initialize LLM (requires valid API key and config)
    config_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        ".config_examples",
        "openai.yaml"
    )
    
    # Check if config exists
    if not os.path.exists(config_path):
        print("Config file not found. Please create a config file first.")
        print(f"Expected location: {config_path}")
        return
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY environment variable not set.")
        print("Please set it to run this example.")
        return
    
    print("Initializing YAMLLM with metrics tracking...")
    llm = LLM(config_path=config_path, api_key=api_key)
    
    # Perform some operations (if you want to test with real API calls)
    # response = llm.chat("Hello, how are you?")
    # print(f"Response: {response}")
    
    # Get metrics summary
    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS SUMMARY")
    print("=" * 60)
    
    summary = llm.get_metrics_summary()
    
    print(f"\nUptime: {summary['uptime_seconds']:.2f} seconds")
    
    print("\nRequests:")
    print(f"  Total: {summary['requests']['total']}")
    print(f"  Avg time: {summary['requests']['avg_time_seconds']:.3f}s")
    
    print("\nLatency:")
    print(f"  Avg first token: {summary['latency']['avg_first_token_ms']:.2f}ms")
    print(f"  P95 first token: {summary['latency']['p95_first_token_ms']:.2f}ms")
    
    print("\nTokens:")
    print(f"  Prompt: {summary['tokens']['prompt']}")
    print(f"  Completion: {summary['tokens']['completion']}")
    print(f"  Total: {summary['tokens']['total']}")
    print(f"  Throughput: {summary['tokens']['throughput_per_second']:.2f} tokens/s")
    
    print("\nTools:")
    print(f"  Executions: {summary['tools']['execution_count']}")
    print(f"  Avg time: {summary['tools']['avg_execution_time_seconds']:.3f}s")
    
    print("\nCache:")
    print(f"  Embedding hit rate: {summary['cache']['embedding_hit_rate_percent']:.2f}%")
    print(f"  Tool def hit rate: {summary['cache']['tool_def_hit_rate_percent']:.2f}%")
    
    # Show Prometheus format
    print("\n" + "=" * 60)
    print("PROMETHEUS METRICS FORMAT")
    print("=" * 60)
    print(llm.get_prometheus_metrics())
    
    # Show cache effectiveness
    print("\n" + "=" * 60)
    print("CACHE ANALYSIS")
    print("=" * 60)
    
    embedding_hit_rate = summary['cache']['embedding_hit_rate_percent']
    tool_def_hit_rate = summary['cache']['tool_def_hit_rate_percent']
    
    if embedding_hit_rate >= 80:
        print("✓ Embedding cache is performing well (≥80% hit rate)")
    else:
        print(f"⚠ Embedding cache could be improved ({embedding_hit_rate:.1f}% < 80%)")
        print("  Consider:")
        print("  - Increasing cache size")
        print("  - Reviewing access patterns")
    
    if tool_def_hit_rate >= 95:
        print("✓ Tool definition cache is performing well (≥95% hit rate)")
    else:
        print(f"⚠ Tool definition cache could be improved ({tool_def_hit_rate:.1f}% < 95%)")
        print("  Consider:")
        print("  - Reducing config changes")
        print("  - Verifying cache invalidation logic")
    
    print("\n" + "=" * 60)
    print("Note: Metrics are tracked automatically and can be accessed")
    print("at any time using llm.get_metrics_summary()")
    print("=" * 60)


if __name__ == "__main__":
    main()
