"""
Example of using AsyncLLM for concurrent requests.

This demonstrates how to use the async interface for better performance
when handling multiple LLM requests.
"""

import asyncio
import time
from yamllm.async_llm import AsyncLLM


async def single_query_example():
    """Example of a single async query."""
    config_path = ".config_examples/openai/basic_config_openai.yaml"
    api_key = "your-api-key"  # In production, use environment variable
    
    async with AsyncLLM(config_path, api_key) as llm:
        response = await llm.query("What is the capital of France?")
        print(f"Response: {response}")


async def streaming_example():
    """Example of async streaming."""
    config_path = ".config_examples/openai_config.yaml"
    api_key = "your-api-key"
    
    async with AsyncLLM(config_path, api_key) as llm:
        print("Streaming response:")
        async for chunk in llm.query_stream("Tell me a short story about a robot."):
            print(chunk, end="", flush=True)
        print("\n")


async def concurrent_queries_example():
    """Example of concurrent queries for better performance."""
    config_path = ".config_examples/openai_config.yaml"
    api_key = "your-api-key"
    
    prompts = [
        "What is the capital of France?",
        "What is the largest planet in our solar system?",
        "Who wrote Romeo and Juliet?",
        "What is the speed of light?",
        "What year did World War II end?"
    ]
    
    async with AsyncLLM(config_path, api_key) as llm:
        # Time concurrent execution
        start_time = time.time()
        responses = await llm.query_many(prompts)
        concurrent_time = time.time() - start_time
        
        print(f"Concurrent execution took: {concurrent_time:.2f} seconds")
        print("\nResponses:")
        for prompt, response in zip(prompts, responses):
            print(f"\nQ: {prompt}")
            print(f"A: {response}")
        
        # Compare with sequential execution
        print("\n" + "="*50 + "\n")
        print("Running same queries sequentially for comparison...")
        
        start_time = time.time()
        sequential_responses = []
        for prompt in prompts:
            response = await llm.query(prompt)
            sequential_responses.append(response)
        sequential_time = time.time() - start_time
        
        print(f"\nSequential execution took: {sequential_time:.2f} seconds")
        print(f"Speedup from concurrent execution: {sequential_time/concurrent_time:.2f}x")


async def mixed_streaming_example():
    """Example of handling multiple streaming requests concurrently."""
    config_path = ".config_examples/openai_config.yaml"
    api_key = "your-api-key"
    
    prompts = [
        "Count from 1 to 5 slowly.",
        "List the primary colors.",
        "Name three programming languages."
    ]
    
    async def stream_with_prefix(llm, prompt, prefix):
        """Helper to stream with a prefix for each line."""
        print(f"\n{prefix}: ", end="")
        async for chunk in llm.query_stream(prompt):
            print(chunk, end="", flush=True)
            if '\n' in chunk:
                print(f"{prefix}: ", end="")
        print()
    
    async with AsyncLLM(config_path, api_key) as llm:
        # Create streaming tasks
        tasks = []
        for i, prompt in enumerate(prompts):
            task = stream_with_prefix(llm, prompt, f"Stream {i+1}")
            tasks.append(task)
        
        # Run all streams concurrently
        print("Running multiple streams concurrently:")
        await asyncio.gather(*tasks)


async def main():
    """Run all examples."""
    print("=== Single Query Example ===")
    await single_query_example()
    
    print("\n=== Streaming Example ===")
    await streaming_example()
    
    print("\n=== Concurrent Queries Example ===")
    await concurrent_queries_example()
    
    print("\n=== Mixed Streaming Example ===")
    await mixed_streaming_example()


if __name__ == "__main__":
    # Run the async examples
    asyncio.run(main())
