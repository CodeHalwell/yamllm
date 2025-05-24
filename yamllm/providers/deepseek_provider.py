"""
DeepSeek provider implementation for YAMLLM.

This module provides an implementation of the BaseProvider interface for DeepSeek.
"""

from typing import Optional

from yamllm.providers.openai_provider import OpenAIProvider


class DeepSeekProvider(OpenAIProvider):
    """
    DeepSeek provider implementation.
    
    This class implements the BaseProvider interface for DeepSeek.
    Since DeepSeek uses an OpenAI-compatible API, it inherits from OpenAIProvider
    and only overrides what's necessary.
    """
    
    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None, **kwargs):
        """
        Initialize the DeepSeek provider.
        
        Args:
            api_key (str): The API key for DeepSeek.
            model (str): The model to use.
            base_url (str, optional): The base URL for the DeepSeek API.
            **kwargs: Additional provider-specific parameters.
        """
        super().__init__(api_key, model, base_url, **kwargs)