"""
OpenRouter provider implementation for unified core providers.

OpenRouter exposes an OpenAI-compatible API across many models at:
  https://openrouter.ai/api/v1

Notes:
- API key via OPENROUTER_API_KEY (or passed explicitly)
- Embeddings are not guaranteed; LLM will fallback to OpenAI embeddings if configured
"""

from typing import Optional, Any

from .openai import OpenAIProvider
from openai import OpenAI


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter provider using the OpenAI-compatible SDK interface."""

    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs: Any):
        # Default endpoint for OpenRouter
        base = base_url or "https://openrouter.ai/api/v1"

        # Extract suggested headers per OpenRouter best practices
        referer = kwargs.get("referer") or kwargs.get("site") or kwargs.get("http_referer")
        title = kwargs.get("title") or kwargs.get("x_title")
        default_headers = kwargs.get("default_headers") or {}
        if referer:
            default_headers["HTTP-Referer"] = referer
        if title:
            default_headers["X-Title"] = title

        # Initialize OpenAI-compatible clients with headers
        self.api_key = api_key
        self.base_url = base
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, default_headers=default_headers or None)
        self.embedding_client = self.client  # embeddings may not be supported; LLM will fallback if needed
