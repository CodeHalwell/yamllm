"""
Compatibility shim for Azure OpenAI provider.

Re-exports the unified core provider to avoid duplication and legacy UI.
Tests may patch `Console` here; provide a harmless placeholder.
"""

# Placeholder for tests that patch this symbol
class Console:  # pragma: no cover - test-only compatibility
    pass

from yamllm.providers.azure_openai import AzureOpenAIProvider  # re-export

__all__ = ["AzureOpenAIProvider", "Console"]

