"""
Async LLM implementation for YAMLLM.

This module provides a fully async LLM interface for better performance
when handling multiple concurrent requests.
"""

import asyncio
from typing import Optional, Dict, Any, List, Callable, AsyncIterator
import logging

from yamllm.core.parser import parse_yaml_config
from yamllm.core.config_validator import ConfigValidator
from yamllm.core.exceptions import ConfigurationError
from yamllm.providers.async_openai import AsyncOpenAIProvider
from yamllm.providers.exceptions import ProviderError
from yamllm.tools.async_manager import AsyncToolManager


class AsyncLLM:
    """
    Async LLM interface for concurrent request handling.
    
    This class provides full async/await support for better performance
    when dealing with multiple LLM requests.
    """
    
    def __init__(self, config_path: str, api_key: str):
        """
        Initialize async LLM.
        
        Args:
            config_path: Path to YAML configuration
            api_key: API key for the provider
        """
        self.config_path = config_path
        self.api_key = api_key
        self.config = self._load_config()
        self.logger = logging.getLogger('yamllm.async')
        
        # Extract config values
        self.provider_name = self.config.provider.name
        self.model = self.config.provider.model
        self.temperature = self.config.model_settings.temperature
        self.max_tokens = self.config.model_settings.max_tokens
        self.top_p = self.config.model_settings.top_p
        self.stop_sequences = self.config.model_settings.stop_sequences
        self.system_prompt = self.config.context.system_prompt
        
        # Initialize components (will be set up in async context)
        self.provider_client = None
        self.tool_manager = None
        
        # Callbacks
        self.stream_callback: Optional[Callable[[str], None]] = None
        self.event_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    
    def _load_config(self):
        """Load and validate configuration."""
        config = parse_yaml_config(self.config_path)
        
        # Validate
        try:
            config_dict = config.model_dump()
        except Exception:
            config_dict = config.dict()
        
        errors = ConfigValidator.validate_config(config_dict)
        if errors:
            raise ConfigurationError(f"Invalid configuration: {'; '.join(errors)}")
        
        return config
    
    async def __aenter__(self):
        """Async context manager entry."""
        # Initialize provider
        if self.provider_name.lower() == "openai":
            self.provider_client = AsyncOpenAIProvider(
                api_key=self.api_key,
                base_url=self.config.provider.base_url
            )
            await self.provider_client.__aenter__()
        else:
            raise ProviderError(
                self.provider_name,
                "Async support not yet implemented for this provider"
            )
        
        # Initialize tool manager if tools are enabled
        if self.config.tools.enabled:
            self.tool_manager = AsyncToolManager(
                timeout=self.config.tools.tool_timeout,
                logger=self.logger
            )
            # Register tools (simplified for demo)
            self._register_tools()
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.provider_client:
            await self.provider_client.__aexit__(exc_type, exc_val, exc_tb)
        
        if self.tool_manager:
            await self.tool_manager.aclose()
    
    def _register_tools(self):
        """Register configured tools (simplified)."""
        # In production, this would register all configured tools
        # For now, just a placeholder
        pass
    
    async def query(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Send an async query to the model.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            Model response
        """
        if not self.provider_client:
            raise ValueError("AsyncLLM must be used within async context manager")
        
        messages = self._prepare_messages(prompt, system_prompt)
        
        # Get response
        response = await self.provider_client.get_completion(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            stop_sequences=self.stop_sequences
        )
        
        # Extract text (simplified)
        if hasattr(response, 'choices') and response.choices:
            return response.choices[0].message.content or ""
        
        return str(response)
    
    async def query_stream(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> AsyncIterator[str]:
        """
        Get a streaming response.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Yields:
            Response chunks
        """
        if not self.provider_client:
            raise ValueError("AsyncLLM must be used within async context manager")
        
        messages = self._prepare_messages(prompt, system_prompt)
        
        # Get streaming response
        async for chunk in self.provider_client.get_streaming_completion(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            stop_sequences=self.stop_sequences
        ):
            # Extract text from chunk
            if hasattr(chunk, 'choices') and chunk.choices:
                if hasattr(chunk.choices[0], 'delta'):
                    text = chunk.choices[0].delta.content
                    if text:
                        yield text
                        
                        # Call callback if set
                        if self.stream_callback:
                            self.stream_callback(text)
    
    async def query_many(
        self, prompts: List[str], system_prompt: Optional[str] = None
    ) -> List[str]:
        """
        Query multiple prompts concurrently.
        
        Args:
            prompts: List of user prompts
            system_prompt: Optional system prompt
            
        Returns:
            List of responses in the same order as prompts
        """
        tasks = []
        for prompt in prompts:
            task = self.query(prompt, system_prompt)
            tasks.append(task)
        
        # Execute all queries concurrently
        responses = await asyncio.gather(*tasks)
        return responses
    
    def _prepare_messages(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Prepare messages for API request."""
        messages = []
        
        # Add system prompt
        if system_prompt or self.system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt or self.system_prompt
            })
        
        # Add user prompt
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        return messages
    
    async def execute_with_tools(
        self, prompt: str, tools: List[str]
    ) -> Dict[str, Any]:
        """
        Execute a prompt with specific tools enabled.
        
        Args:
            prompt: User prompt
            tools: List of tool names to use
            
        Returns:
            Dict with 'response' and 'tool_results'
        """
        if not self.tool_manager:
            raise ValueError("Tools not enabled in configuration")
        
        # This is a simplified implementation
        # In production, this would handle the full tool calling flow
        
        # First, get the model's response with tools
        messages = self._prepare_messages(prompt)
        tool_definitions = []  # Would get from tool_manager
        
        response = await self.provider_client.get_completion(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            tools=tool_definitions if tool_definitions else None
        )
        
        # Check for tool calls and execute them
        # This is simplified - real implementation would handle tool calls properly
        
        return {
            "response": self._extract_text(response),
            "tool_results": []
        }
    
    def _extract_text(self, response) -> str:
        """Extract text from response."""
        if hasattr(response, 'choices') and response.choices:
            return response.choices[0].message.content or ""
        return str(response)