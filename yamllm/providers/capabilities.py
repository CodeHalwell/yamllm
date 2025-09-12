"""
Provider capability registry.

Defines a simple, centralized map of supported features per provider so that
the core can gracefully enable/disable behaviors like tool calling, streaming,
and embeddings across providers without brittle runtime checks.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ProviderCapabilities:
    supports_tools: bool = False
    supports_streaming: bool = False
    supports_embeddings: bool = False
    max_tokens: int = 4096
    tool_calling_format: str = "openai"  # "openai", "anthropic", "google", etc.


# Known providers and their best-effort capability flags. These are conservative
# defaults intended to prevent unsupported flows rather than over-promise.
PROVIDER_CAPABILITIES: Dict[str, ProviderCapabilities] = {
    # OpenAI-compatible implementations
    "openai": ProviderCapabilities(
        supports_tools=True,
        supports_streaming=True,
        supports_embeddings=True,
        max_tokens=128_000,
        tool_calling_format="openai",
    ),
    "azure_openai": ProviderCapabilities(
        supports_tools=True,
        supports_streaming=True,
        supports_embeddings=True,
        max_tokens=128_000,
        tool_calling_format="openai",
    ),
    "openrouter": ProviderCapabilities(
        supports_tools=True,
        supports_streaming=True,
        supports_embeddings=False,  # depends on routed model; default to False
        max_tokens=128_000,
        tool_calling_format="openai",
    ),
    # Other providers with custom formats
    "anthropic": ProviderCapabilities(
        supports_tools=True,
        supports_streaming=True,
        supports_embeddings=False,
        max_tokens=200_000,
        tool_calling_format="anthropic",
    ),
    "google": ProviderCapabilities(
        supports_tools=True,
        supports_streaming=True,
        supports_embeddings=True,
        max_tokens=1_048_576,
        tool_calling_format="google",
    ),
    "mistral": ProviderCapabilities(
        supports_tools=True,
        supports_streaming=True,
        supports_embeddings=True,
        max_tokens=32_000,
        tool_calling_format="openai",
    ),
    "azure_foundry": ProviderCapabilities(
        supports_tools=True,  # API parity depends on model; assume True for chat
        supports_streaming=True,
        supports_embeddings=True,
        max_tokens=128_000,
        tool_calling_format="openai",
    ),
    # If "deepseek" is used via OpenAI-compatible APIs, keep default elsewhere
    "deepseek": ProviderCapabilities(
        supports_tools=True,
        supports_streaming=True,
        supports_embeddings=False,
        max_tokens=128_000,
        tool_calling_format="openai",
    ),
}


def get_provider_capabilities(provider_name: str) -> ProviderCapabilities:
    """Return capabilities for the given provider name (case-insensitive)."""
    return PROVIDER_CAPABILITIES.get((provider_name or "").lower(), ProviderCapabilities())

