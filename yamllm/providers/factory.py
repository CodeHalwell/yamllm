"""
Provider factory for unified core providers.

Lazily imports providers to avoid import-time dependency errors for optional
providers (e.g., Google SDK types) during test collection or restricted envs.
"""

from typing import Optional, Any, Type, Dict
import importlib

from .base import BaseProvider
from .async_base import AsyncBaseProvider


def _load_class(path: str):
    module_name, _, class_name = path.rpartition(".")
    if not module_name:
        raise ValueError(f"Invalid class path: {path}")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


class ProviderFactory:
    """
    Factory for creating provider instances backed by core providers.
    """

    _SYNC_CLASS_PATHS: Dict[str, str] = {
        "openai": "yamllm.providers.openai.OpenAIProvider",
        "anthropic": "yamllm.providers.anthropic.AnthropicProvider",
        "azure_openai": "yamllm.providers.azure_openai.AzureOpenAIProvider",
        "azure_foundry": "yamllm.providers.azure_foundry.AzureFoundryProvider",
        "google": "yamllm.providers.google.GoogleGeminiProvider",
        "mistral": "yamllm.providers.mistral.MistralProvider",
        "openrouter": "yamllm.providers.openrouter.OpenRouterProvider",
        "deepseek": "yamllm.providers.openai.OpenAIProvider",  # OpenAI-compatible
    }
    # Back-compat alias used by some tests
    _MAP = _SYNC_CLASS_PATHS
    
    _ASYNC_CLASS_PATHS: Dict[str, str] = {
        "openai": "yamllm.providers.async_openai.AsyncOpenAIProvider",
        "anthropic": "yamllm.providers.async_anthropic.AsyncAnthropicProvider",
        "google": "yamllm.providers.async_google.AsyncGoogleProvider",
        "mistral": "yamllm.providers.async_mistral.AsyncMistralProvider",
        "azure_openai": "yamllm.providers.async_azure_openai.AsyncAzureOpenAIProvider",
        "openrouter": "yamllm.providers.async_openai.AsyncOpenAIProvider",  # OpenAI-compatible
        "deepseek": "yamllm.providers.async_openai.AsyncOpenAIProvider",  # OpenAI-compatible
    }

    @classmethod
    def create_provider(
        cls,
        provider_name: str,
        *,
        api_key: str,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseProvider:
        name = (provider_name or "").lower()
        if name not in cls._SYNC_CLASS_PATHS:
            raise ValueError(
                f"Unknown provider: {provider_name}. Available providers: {', '.join(cls._SYNC_CLASS_PATHS.keys())}"
            )
        
        # Special handling for certain providers
        if name == "deepseek" and not base_url:
            base_url = "https://api.deepseek.com/v1"
        elif name == "openrouter" and not base_url:
            base_url = "https://openrouter.ai/api/v1"
        
        provider_cls = _load_class(cls._SYNC_CLASS_PATHS[name])
        return provider_cls(api_key=api_key, base_url=base_url, **kwargs)
    
    @classmethod
    def create_async_provider(
        cls,
        provider_name: str,
        *,
        api_key: str,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncBaseProvider:
        """
        Create an async provider instance.
        
        Args:
            provider_name: Name of the provider
            api_key: API key for the provider
            base_url: Optional base URL
            **kwargs: Additional provider-specific parameters
            
        Returns:
            AsyncBaseProvider: Async provider instance
            
        Raises:
            ValueError: If provider doesn't support async
        """
        name = (provider_name or "").lower()
        if name not in cls._ASYNC_CLASS_PATHS:
            raise ValueError(
                f"Async support not available for provider '{provider_name}'. "
                f"Available async providers: {', '.join(cls._ASYNC_CLASS_PATHS.keys())}"
            )
        
        # Special handling for certain providers
        if name == "deepseek" and not base_url:
            base_url = "https://api.deepseek.com/v1"
        elif name == "openrouter" and not base_url:
            base_url = "https://openrouter.ai/api/v1"
        
        provider_cls = _load_class(cls._ASYNC_CLASS_PATHS[name])
        return provider_cls(api_key=api_key, base_url=base_url, **kwargs)
    
    @classmethod
    def register_provider(
        cls,
        name: str,
        provider_class: Type[BaseProvider],
        async_provider_class: Optional[Type[AsyncBaseProvider]] = None
    ):
        """
        Register a custom provider.
        
        Args:
            name: Provider name
            provider_class: Sync provider class
            async_provider_class: Optional async provider class
        """
        # Register eagerly for already-imported classes; otherwise, store path
        key = name.lower()
        cls._SYNC_CLASS_PATHS[key] = f"{provider_class.__module__}.{provider_class.__name__}"
        if async_provider_class:
            cls._ASYNC_CLASS_PATHS[key] = f"{async_provider_class.__module__}.{async_provider_class.__name__}"
    
    @classmethod
    def list_providers(cls) -> list:
        """List available providers."""
        return list(cls._SYNC_CLASS_PATHS.keys())
    
    @classmethod
    def list_async_providers(cls) -> list:
        """List providers with async support."""
        return list(cls._ASYNC_CLASS_PATHS.keys())
    
    @classmethod
    def supports_async(cls, provider_name: str) -> bool:
        """Check if a provider supports async operations."""
        return provider_name.lower() in cls._ASYNC_CLASS_PATHS
