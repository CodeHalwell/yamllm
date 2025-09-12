"""
Integration tests for all LLM providers.

These tests verify that each provider works correctly with real API calls.
They require valid API keys to be set in environment variables.
"""

import os
import pytest
import asyncio
from typing import Dict, Any

from yamllm.providers.factory import ProviderFactory


# Skip integration tests if no API keys are configured
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
MISTRAL_KEY = os.getenv("MISTRAL_API_KEY")
AZURE_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

# Test configuration for each provider
PROVIDER_CONFIGS = {
    "openai": {
        "api_key": OPENAI_KEY,
        "model": "gpt-3.5-turbo",
        "skip": not OPENAI_KEY,
        "skip_reason": "OPENAI_API_KEY not set"
    },
    "anthropic": {
        "api_key": ANTHROPIC_KEY,
        "model": "claude-3-haiku-20240307",
        "skip": not ANTHROPIC_KEY,
        "skip_reason": "ANTHROPIC_API_KEY not set"
    },
    "google": {
        "api_key": GOOGLE_KEY,
        "model": "gemini-pro",
        "skip": not GOOGLE_KEY,
        "skip_reason": "GOOGLE_API_KEY not set"
    },
    "mistral": {
        "api_key": MISTRAL_KEY,
        "model": "mistral-tiny",
        "skip": not MISTRAL_KEY,
        "skip_reason": "MISTRAL_API_KEY not set"
    },
    "azure_openai": {
        "api_key": AZURE_KEY,
        "model": "gpt-35-turbo",
        "base_url": AZURE_ENDPOINT,
        "skip": not (AZURE_KEY and AZURE_ENDPOINT),
        "skip_reason": "AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT not set",
        "extra_kwargs": {"deployment_name": "gpt-35-turbo"}
    },
    "openrouter": {
        "api_key": OPENROUTER_KEY,
        "model": "openai/gpt-3.5-turbo",
        "skip": not OPENROUTER_KEY,
        "skip_reason": "OPENROUTER_API_KEY not set"
    }
}


class TestProvidersIntegration:
    """Integration tests for sync providers."""
    
    @pytest.mark.parametrize("provider_name,config", PROVIDER_CONFIGS.items())
    def test_provider_basic_completion(self, provider_name: str, config: Dict[str, Any]):
        """Test basic completion for each provider."""
        if config["skip"]:
            pytest.skip(config["skip_reason"])
        
        # Create provider
        provider = ProviderFactory.create_provider(
            provider_name,
            api_key=config["api_key"],
            base_url=config.get("base_url"),
            **config.get("extra_kwargs", {})
        )
        
        # Test messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello, World!' and nothing else."}
        ]
        
        # Get completion
        response = provider.get_completion(
            messages=messages,
            model=config["model"],
            temperature=0.5,
            max_tokens=50,
            top_p=1.0
        )
        
        # Verify response
        assert response is not None
        
        # Extract text based on provider
        text = self._extract_text(response, provider_name)
        assert text is not None
        assert len(text) > 0
        assert "hello" in text.lower() or "world" in text.lower()
    
    @pytest.mark.parametrize("provider_name,config", PROVIDER_CONFIGS.items())
    def test_provider_streaming_completion(self, provider_name: str, config: Dict[str, Any]):
        """Test streaming completion for each provider."""
        if config["skip"]:
            pytest.skip(config["skip_reason"])
        
        # Azure Foundry doesn't support streaming yet
        if provider_name == "azure_foundry":
            pytest.skip("Azure Foundry doesn't support streaming")
        
        # Create provider
        provider = ProviderFactory.create_provider(
            provider_name,
            api_key=config["api_key"],
            base_url=config.get("base_url"),
            **config.get("extra_kwargs", {})
        )
        
        # Test messages
        messages = [
            {"role": "user", "content": "Count from 1 to 5."}
        ]
        
        # Get streaming completion
        stream = provider.get_streaming_completion(
            messages=messages,
            model=config["model"],
            temperature=0.5,
            max_tokens=100,
            top_p=1.0
        )
        
        # Collect chunks
        chunks = []
        for chunk in stream:
            chunk_text = self._extract_chunk_text(chunk, provider_name)
            if chunk_text:
                chunks.append(chunk_text)
        
        # Verify we got multiple chunks
        assert len(chunks) > 0
        full_text = "".join(chunks)
        assert len(full_text) > 0
    
    @pytest.mark.parametrize("provider_name,config", [
        (name, conf) for name, conf in PROVIDER_CONFIGS.items() 
        if name in ["openai", "azure_openai", "mistral", "anthropic"]
    ])
    def test_provider_tool_calling(self, provider_name: str, config: Dict[str, Any]):
        """Test tool calling for providers that support it."""
        if config["skip"]:
            pytest.skip(config["skip_reason"])
        
        # Create provider
        provider = ProviderFactory.create_provider(
            provider_name,
            api_key=config["api_key"],
            base_url=config.get("base_url"),
            **config.get("extra_kwargs", {})
        )
        
        # Define a simple tool
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            }
        }]
        
        # Test message that should trigger tool use
        messages = [
            {"role": "user", "content": "What's the weather in New York?"}
        ]
        
        # Get completion with tools
        response = provider.get_completion(
            messages=messages,
            model=config["model"],
            temperature=0.5,
            max_tokens=100,
            top_p=1.0,
            tools=tools
        )
        
        # Check if tool was called
        tool_calls = self._extract_tool_calls(response, provider_name)
        if tool_calls:  # Some models might not always call tools
            assert len(tool_calls) > 0
            assert tool_calls[0]["function"]["name"] == "get_weather"
            assert "location" in tool_calls[0]["function"].get("arguments", "{}")
    
    def _extract_text(self, response: Any, provider_name: str) -> str:
        """Extract text from provider response."""
        if provider_name in ["openai", "azure_openai", "mistral", "openrouter"]:
            if hasattr(response, 'choices') and response.choices:
                return response.choices[0].message.content or ""
        elif provider_name == "anthropic":
            if isinstance(response, dict) and "content" in response:
                return response["content"]
        elif provider_name == "google":
            if hasattr(response, 'text'):
                return response.text
            elif isinstance(response, dict) and "text" in response:
                return response["text"]
        
        return str(response)
    
    def _extract_chunk_text(self, chunk: Any, provider_name: str) -> str:
        """Extract text from streaming chunk."""
        if provider_name in ["openai", "azure_openai", "mistral", "openrouter"]:
            if hasattr(chunk, 'choices') and chunk.choices:
                if hasattr(chunk.choices[0], 'delta'):
                    return chunk.choices[0].delta.content or ""
        elif provider_name == "anthropic":
            if isinstance(chunk, dict) and chunk.get("type") == "content_block_delta":
                return chunk.get("delta", {}).get("text", "")
        elif provider_name == "google":
            if hasattr(chunk, 'text'):
                return chunk.text or ""
        
        return ""
    
    def _extract_tool_calls(self, response: Any, provider_name: str) -> list:
        """Extract tool calls from response."""
        if provider_name in ["openai", "azure_openai", "mistral", "openrouter"]:
            if hasattr(response, 'choices') and response.choices:
                message = response.choices[0].message
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    provider = ProviderFactory.create_provider(
                        provider_name, api_key="dummy"
                    )
                    return provider.format_tool_calls(message.tool_calls)
        elif provider_name == "anthropic":
            if isinstance(response, dict) and "tool_calls" in response:
                return response["tool_calls"]
        
        return []


class TestAsyncProvidersIntegration:
    """Integration tests for async providers."""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("provider_name,config", [
        (name, conf) for name, conf in PROVIDER_CONFIGS.items()
        if ProviderFactory.supports_async(name)
    ])
    async def test_async_provider_basic_completion(self, provider_name: str, config: Dict[str, Any]):
        """Test async basic completion for each provider."""
        if config["skip"]:
            pytest.skip(config["skip_reason"])
        
        # Create async provider
        provider = ProviderFactory.create_async_provider(
            provider_name,
            api_key=config["api_key"],
            base_url=config.get("base_url"),
            **config.get("extra_kwargs", {})
        )
        
        async with provider:
            # Test messages
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello, Async!' and nothing else."}
            ]
            
            # Get completion
            response = await provider.get_completion(
                messages=messages,
                model=config["model"],
                temperature=0.5,
                max_tokens=50,
                top_p=1.0
            )
            
            # Verify response
            assert response is not None
            text = self._extract_async_text(response, provider_name)
            assert text is not None
            assert len(text) > 0
            assert "hello" in text.lower() or "async" in text.lower()
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("provider_name,config", [
        (name, conf) for name, conf in PROVIDER_CONFIGS.items()
        if ProviderFactory.supports_async(name)
    ])
    async def test_async_provider_streaming(self, provider_name: str, config: Dict[str, Any]):
        """Test async streaming for each provider."""
        if config["skip"]:
            pytest.skip(config["skip_reason"])
        
        # Create async provider
        provider = ProviderFactory.create_async_provider(
            provider_name,
            api_key=config["api_key"],
            base_url=config.get("base_url"),
            **config.get("extra_kwargs", {})
        )
        
        async with provider:
            # Test messages
            messages = [
                {"role": "user", "content": "List three colors."}
            ]
            
            # Get streaming completion
            chunks = []
            async for chunk in provider.get_streaming_completion(
                messages=messages,
                model=config["model"],
                temperature=0.5,
                max_tokens=100,
                top_p=1.0
            ):
                if isinstance(chunk, str):
                    chunks.append(chunk)
                else:
                    chunk_text = self._extract_async_chunk_text(chunk, provider_name)
                    if chunk_text:
                        chunks.append(chunk_text)
            
            # Verify we got chunks
            assert len(chunks) > 0
            full_text = "".join(chunks)
            assert len(full_text) > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_async_requests(self):
        """Test concurrent requests across multiple providers."""
        # Get available providers with API keys
        available_providers = []
        for name, config in PROVIDER_CONFIGS.items():
            if not config["skip"] and ProviderFactory.supports_async(name):
                available_providers.append((name, config))
        
        if len(available_providers) < 2:
            pytest.skip("Need at least 2 providers with API keys for concurrent test")
        
        # Use first two available providers
        providers_to_test = available_providers[:2]
        
        async def query_provider(provider_name: str, config: Dict[str, Any]):
            """Query a single provider."""
            provider = ProviderFactory.create_async_provider(
                provider_name,
                api_key=config["api_key"],
                base_url=config.get("base_url"),
                **config.get("extra_kwargs", {})
            )
            
            async with provider:
                response = await provider.get_completion(
                    messages=[{"role": "user", "content": f"Say '{provider_name} works!'"}],
                    model=config["model"],
                    temperature=0.5,
                    max_tokens=50,
                    top_p=1.0
                )
                
                return provider_name, self._extract_async_text(response, provider_name)
        
        # Run queries concurrently
        tasks = [
            query_provider(name, config)
            for name, config in providers_to_test
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify all providers responded
        assert len(results) == len(providers_to_test)
        for provider_name, text in results:
            assert text is not None
            assert len(text) > 0
            assert provider_name.lower() in text.lower() or "works" in text.lower()
    
    def _extract_async_text(self, response: Any, provider_name: str) -> str:
        """Extract text from async response."""
        if provider_name in ["openai", "azure_openai", "mistral", "openrouter", "deepseek"]:
            if hasattr(response, 'choices') and response.choices:
                return response.choices[0].message.content or ""
        elif provider_name == "anthropic":
            if isinstance(response, dict) and "content" in response:
                return response["content"]
        elif provider_name == "google":
            if isinstance(response, dict) and "text" in response:
                return response["text"]
        
        return str(response)
    
    def _extract_async_chunk_text(self, chunk: Any, provider_name: str) -> str:
        """Extract text from async streaming chunk."""
        if provider_name in ["openai", "azure_openai", "mistral", "openrouter", "deepseek"]:
            if hasattr(chunk, 'choices') and chunk.choices:
                if hasattr(chunk.choices[0], 'delta'):
                    return chunk.choices[0].delta.content or ""
        elif provider_name == "anthropic":
            # Anthropic async yields strings directly
            if isinstance(chunk, str):
                return chunk
        elif provider_name == "google":
            # Google async yields strings directly  
            if isinstance(chunk, str):
                return chunk
        
        return ""


# Performance comparison tests
class TestProviderPerformance:
    """Compare performance between sync and async providers."""
    
    @pytest.mark.parametrize("provider_name,config", [
        (name, conf) for name, conf in PROVIDER_CONFIGS.items()
        if ProviderFactory.supports_async(name) and name == "openai"  # Test with one provider
    ])
    def test_sync_vs_async_performance(self, provider_name: str, config: Dict[str, Any]):
        """Compare sync vs async performance for multiple requests."""
        if config["skip"]:
            pytest.skip(config["skip_reason"])
        
        import time
        
        # Number of requests to make
        num_requests = 3
        messages = [{"role": "user", "content": "Say 'test' and nothing else."}]
        
        # Test sync performance
        sync_provider = ProviderFactory.create_provider(
            provider_name,
            api_key=config["api_key"],
            base_url=config.get("base_url"),
            **config.get("extra_kwargs", {})
        )
        
        sync_start = time.time()
        for _ in range(num_requests):
            sync_provider.get_completion(
                messages=messages,
                model=config["model"],
                temperature=0.5,
                max_tokens=10,
                top_p=1.0
            )
        sync_time = time.time() - sync_start
        
        # Test async performance
        async def run_async_test():
            provider = ProviderFactory.create_async_provider(
                provider_name,
                api_key=config["api_key"],
                base_url=config.get("base_url"),
                **config.get("extra_kwargs", {})
            )
            
            async with provider:
                tasks = []
                for _ in range(num_requests):
                    task = provider.get_completion(
                        messages=messages,
                        model=config["model"],
                        temperature=0.5,
                        max_tokens=10,
                        top_p=1.0
                    )
                    tasks.append(task)
                
                await asyncio.gather(*tasks)
        
        async_start = time.time()
        asyncio.run(run_async_test())
        async_time = time.time() - async_start
        
        # Async should be faster for multiple requests
        print(f"\nProvider: {provider_name}")
        print(f"Sync time for {num_requests} requests: {sync_time:.2f}s")
        print(f"Async time for {num_requests} requests: {async_time:.2f}s")
        print(f"Speedup: {sync_time/async_time:.2f}x")
        
        # Async should be at least somewhat faster
        assert async_time < sync_time * 1.1  # Allow 10% margin