"""
DeepSeek provider implementation for YAMLLM.

This module provides an implementation of the BaseProvider interface for DeepSeek.
DeepSeek uses an OpenAI-compatible API with some specific optimizations.
"""

from typing import Dict, List, Any, Optional
from openai import OpenAI

from yamllm.providers.openai_provider import OpenAIProvider


class DeepSeekProvider(OpenAIProvider):
    """
    DeepSeek provider implementation.
    
    This class implements the BaseProvider interface for DeepSeek.
    Since DeepSeek uses an OpenAI-compatible API, it inherits from OpenAIProvider
    and only overrides what's necessary for DeepSeek-specific optimizations.
    
    Features:
    - Support for custom request headers to optimize performance
    - Optional request caching for improved performance
    - Documentation about embedding model limitations
    """
    
    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None, **kwargs):
        """
        Initialize the DeepSeek provider.
        
        Args:
            api_key (str): The API key for DeepSeek.
            model (str): The model to use.
            base_url (str, optional): The base URL for the DeepSeek API.
            **kwargs: Additional provider-specific parameters.
                - headers (Dict[str, str], optional): Custom headers for API requests.
                - cache_enabled (bool, optional): Enable request caching if available.
                - cache_ttl (int, optional): Time-to-live for cached requests in seconds.
        """
        # Extract DeepSeek-specific parameters
        self.custom_headers = kwargs.pop('headers', {})
        self.cache_enabled = kwargs.pop('cache_enabled', False)
        self.cache_ttl = kwargs.pop('cache_ttl', 3600)  # Default 1 hour
        
        # Initialize the parent class
        super().__init__(api_key, model, base_url, **kwargs)
        
        # Reinitialize OpenAI client with custom headers if provided
        if self.custom_headers:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                default_headers=self.custom_headers
            )
    
    def prepare_completion_params(self, messages: List[dict], temperature: float, max_tokens: int, 
                                 top_p: float, stop_sequences: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Prepare completion parameters for DeepSeek's API.
        
        Override to add any DeepSeek-specific parameters or headers.
        
        Args:
            messages (List[dict]): The messages to send to the model.
            temperature (float): The temperature parameter for the model.
            max_tokens (int): The maximum number of tokens to generate.
            top_p (float): The top_p parameter for the model.
            stop_sequences (List[str], optional): Sequences that will stop generation.
            
        Returns:
            Dict[str, Any]: The parameters for the API request.
        """
        # Get base parameters from OpenAIProvider
        params = super().prepare_completion_params(messages, temperature, max_tokens, top_p, stop_sequences)
        
        # Add DeepSeek-specific parameters if needed
        if self.cache_enabled:
            # If caching is enabled, we add this to our custom headers
            # Note: The actual caching implementation depends on DeepSeek's API support
            self.custom_headers['X-DeepSeek-Cache'] = 'true'
            self.custom_headers['X-DeepSeek-Cache-TTL'] = str(self.cache_ttl)
        
        return params
    
    def create_embedding(self, text: str) -> bytes:
        """
        Create an embedding for the given text using DeepSeek's API.
        
        Note that DeepSeek may not support embeddings through their OpenAI-compatible API,
        or may use different embedding models. This implementation falls back to OpenAI's
        embedding model, which may not be optimal for DeepSeek.
        
        Args:
            text (str): The text to create an embedding for.
            
        Returns:
            bytes: The embedding as bytes.
        """
        try:
            # Call the parent implementation for now
            # This uses OpenAI's embedding model, which may not be optimal for DeepSeek
            return super().create_embedding(text)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating embedding with DeepSeek: {str(e)}")
                self.logger.warning("DeepSeek may not support embeddings via OpenAI-compatible API")
            raise Exception(f"Error creating embedding with DeepSeek: {str(e)}")