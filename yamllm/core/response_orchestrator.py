"""
Response orchestration component for YAMLLM.

This module handles coordinating responses between the provider,
tools, and memory management.
"""

from typing import Dict, List, Any, Optional, Callable
import logging


class ResponseOrchestrator:
    """
    Coordinates response generation across providers, tools, and memory.
    
    This class extracts the response orchestration logic from the main
    LLM class for better separation of concerns.
    """
    
    def __init__(
        self,
        provider_client: Any,
        provider_name: str,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop_sequences: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize response orchestrator.
        
        Args:
            provider_client: The provider client instance
            provider_name: Name of the provider (e.g., 'openai')
            model: Model name
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            stop_sequences: Optional stop sequences
            logger: Optional logger instance
        """
        self.provider_client = provider_client
        self.provider_name = provider_name
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.stop_sequences = stop_sequences or []
        self.logger = logger or logging.getLogger('yamllm.orchestrator')
        
        # Usage tracking
        self._last_usage: Optional[Dict[str, int]] = None
        self._total_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    
    def get_non_streaming_response(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Get a non-streaming response from the provider.
        
        Args:
            messages: List of messages to send
            tools: Optional tool definitions
            
        Returns:
            The response text
        """
        try:
            response = self.provider_client.get_completion(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                stop=self.stop_sequences if self.stop_sequences else None,
                tools=tools
            )
            
            # Track usage
            self._update_usage(response)
            
            # Extract text
            return self._extract_text_from_response(response)
            
        except Exception as e:
            self.logger.error(f"Error getting non-streaming response: {e}")
            raise
    
    def _extract_text_from_response(self, response) -> str:
        """Extract text from a non-streaming response."""
        if self.provider_name.lower() in ("openai", "azure_openai", "mistral", "openrouter", "deepseek"):
            if hasattr(response, 'choices') and response.choices:
                return response.choices[0].message.content or ""
        elif self.provider_name.lower() == "anthropic":
            if isinstance(response, dict):
                content = response.get("content", [])
                if content and isinstance(content[0], dict):
                    return content[0].get("text", "")
        elif self.provider_name.lower() == "google":
            if hasattr(response, 'text'):
                return response.text
        
        return str(response)
    
    def _update_usage(self, response):
        """Update usage statistics from response."""
        try:
            usage = None
            if hasattr(response, 'usage'):
                usage = response.usage
            elif isinstance(response, dict) and 'usage' in response:
                usage = response['usage']
            
            if usage:
                prompt_tokens = getattr(usage, 'prompt_tokens', 0) if hasattr(usage, 'prompt_tokens') else usage.get('prompt_tokens', 0)
                completion_tokens = getattr(usage, 'completion_tokens', 0) if hasattr(usage, 'completion_tokens') else usage.get('completion_tokens', 0)
                total_tokens = getattr(usage, 'total_tokens', 0) if hasattr(usage, 'total_tokens') else usage.get('total_tokens', 0)
                
                self._last_usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
                
                self._total_usage["prompt_tokens"] += prompt_tokens
                self._total_usage["completion_tokens"] += completion_tokens
                self._total_usage["total_tokens"] += total_tokens
        except Exception as e:
            self.logger.debug(f"Could not update usage: {e}")
    
    def get_last_usage(self) -> Optional[Dict[str, int]]:
        """Get usage statistics from last request."""
        return self._last_usage
    
    def get_total_usage(self) -> Dict[str, int]:
        """Get total usage statistics."""
        return self._total_usage.copy()
