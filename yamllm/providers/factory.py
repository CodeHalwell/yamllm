"""
Provider factory for YAMLLM.

This module provides a factory for creating provider instances.
"""

from typing import Optional

from yamllm.providers.base import BaseProvider
from yamllm.providers.openai_provider import OpenAIProvider
from yamllm.providers.deepseek_provider import DeepSeekProvider
from yamllm.providers.mistral_provider import MistralProvider
from yamllm.providers.google_provider import GoogleGeminiProvider
from yamllm.providers.azure_openai_provider import AzureOpenAIProvider
from yamllm.providers.azure_foundry_provider import AzureFoundryProvider


class ProviderFactory:
    """
    Factory for creating provider instances.
    
    This class provides a simple factory for creating provider instances
    based on the provider name.
    """
    
    @staticmethod
    def create_provider(provider_name: str, api_key: str, model: str, 
                       base_url: Optional[str] = None, **kwargs) -> BaseProvider:
        """
        Create a provider instance.
        
        Args:
            provider_name (str): The name of the provider.
            api_key (str): The API key for the provider.
            model (str): The model to use.
            base_url (str, optional): The base URL for the provider's API.
            **kwargs: Additional provider-specific parameters.
            
        Returns:
            BaseProvider: A provider instance.
            
        Raises:
            ValueError: If the provider name is not recognized.
        """
        provider_map = {
            "openai": OpenAIProvider,
            "deepseek": DeepSeekProvider,
            "mistral": MistralProvider,
            "google": GoogleGeminiProvider,
            "azure_openai": AzureOpenAIProvider,
            "azure_foundry": AzureFoundryProvider
        }
        
        if provider_name.lower() not in provider_map:
            raise ValueError(f"Unknown provider: {provider_name}. Available providers: {', '.join(provider_map.keys())}")
        
        provider_class = provider_map[provider_name.lower()]
        return provider_class(api_key, model, base_url, **kwargs)