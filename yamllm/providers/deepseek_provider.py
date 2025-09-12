"""
Compatibility shim for DeepSeek provider.

DeepSeek uses an OpenAI-compatible API; expose a provider that inherits from the
unified core OpenAI provider for compatibility with existing imports/tests.
"""

from yamllm.providers.openai import OpenAIProvider as _CoreOpenAIProvider


class DeepSeekProvider(_CoreOpenAIProvider):
    pass

__all__ = ["DeepSeekProvider"]

